package com.micaftic.morpher.resource;

import com.micaftic.morpher.resource.pojo.RawYsmModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Converts a Bedrock {@code poly_mesh} (Blockbench mesh export / glTF-converted models) into the
 * quad based {@link RawYsmModel.RawCube} structures the renderer understands.
 *
 * <p>The YSM render pipeline only knows 4 vertex faces. Every triangle is therefore emitted as its
 * own single-face cube whose 4th vertex repeats the 3rd one (same trick that
 * {@code BBToRawConverter} uses for {@code .bbmodel} meshes), quads stay quads and larger polygons
 * are fan-triangulated.
 *
 * <p>Coordinate conventions are identical to what {@link YsmGeometryParsing} does for cubes:
 * {@code x' = -x / 16, y' = y / 16, z' = z / 16}. Mirroring X flips the winding, so the vertex
 * order is reversed to keep counter-clockwise = front face, and the face normal is recomputed from
 * the final geometry (poly_mesh files very often carry all-zero normals).
 *
 * <p>UVs: with {@code normalized_uvs = true} the V axis has its origin at the bottom
 * (Blockbench / OpenGL style), so {@code v' = 1 - v}. Pixel UVs are divided by the texture size.
 *
 * <p>Shading: every face also gets smooth per-vertex normals ({@code RawFace.vertexNormals}). Faces that share a
 * position are averaged (corner-angle weighted) as long as their face normals differ by at most 60 degrees, so
 * curved surfaces look smooth while genuinely hard edges stay crisp.
 *
 * <p>Pure Java, no Minecraft or Gson dependency, so it can be unit tested on the JVM.
 */
public final class PolyMeshBaker {

    private static final float MIN_NORMAL_LEN = 1e-9f;
    /** cos(60 deg): neighbouring faces steeper than this keep a hard edge. */
    private static final float CREASE_COS = 0.5f;

    private static final class Face {
        float[][] p;   // 4 positions (3 repeated for a triangle)
        float[][] t;   // 4 uvs
        boolean tri;
        float[] n;     // flat face normal
    }

    private record PosKey(int x, int y, int z) {
        static PosKey of(float[] p) {
            return new PosKey(Math.round(p[0] * 100000f), Math.round(p[1] * 100000f), Math.round(p[2] * 100000f));
        }
    }

    private PolyMeshBaker() {
    }

    /**
     * @param positions    vertex positions in model space (Bedrock units, 16 = 1 block)
     * @param uvs          uv table, may be empty
     * @param polys        {@code polys[i][j] = {positionIndex, normalIndex, uvIndex}}
     * @param normalizedUvs {@code normalized_uvs} flag of the poly_mesh
     */
    public static List<RawYsmModel.RawCube> bake(float[][] positions, float[][] uvs, int[][][] polys,
                                                 boolean normalizedUvs, float texW, float texH) {
        List<Face> out = new ArrayList<>(polys.length);
        // exact duplicate faces (same vertices, same UVs, same winding) are dropped: they only cost memory and z-fight
        Set<Long> seen = new HashSet<>(polys.length * 2);
        float tw = Math.max(1f, texW);
        float th = Math.max(1f, texH);

        for (int[][] poly : polys) {
            if (poly == null || poly.length < 3) continue;
            int n = poly.length;
            float[][] pos = new float[n][];
            float[][] uv = new float[n][];
            boolean valid = true;

            // reversed order: mirroring X flips the winding
            for (int i = 0; i < n && valid; i++) {
                int[] vtx = poly[n - 1 - i];
                int pi = vtx != null && vtx.length > 0 ? vtx[0] : -1;
                int ui = vtx != null && vtx.length > 2 ? vtx[2] : -1;
                if (pi < 0 || pi >= positions.length || positions[pi] == null || positions[pi].length < 3) {
                    valid = false;
                    break;
                }
                float[] p = positions[pi];
                pos[i] = new float[]{-p[0] / 16f, p[1] / 16f, p[2] / 16f};

                float u = 0f, v = 0f;
                if (ui >= 0 && ui < uvs.length && uvs[ui] != null && uvs[ui].length >= 2) {
                    u = uvs[ui][0];
                    v = uvs[ui][1];
                    if (normalizedUvs) {
                        v = 1f - v;
                    } else {
                        u /= tw;
                        v /= th;
                    }
                }
                uv[i] = new float[]{u, v};
            }
            if (!valid) continue;

            if (n == 3) {
                emit(out, seen, pos[0], pos[1], pos[2], pos[2], uv[0], uv[1], uv[2], uv[2]);
            } else if (n == 4) {
                emit(out, seen, pos[0], pos[1], pos[2], pos[3], uv[0], uv[1], uv[2], uv[3]);
            } else {
                for (int i = 1; i + 1 < n; i++) {
                    emit(out, seen, pos[0], pos[i], pos[i + 1], pos[i + 1], uv[0], uv[i], uv[i + 1], uv[i + 1]);
                }
            }
        }
        return finish(out);
    }

    private static void emit(List<Face> out, Set<Long> seen,
                             float[] p0, float[] p1, float[] p2, float[] p3,
                             float[] t0, float[] t1, float[] t2, float[] t3) {
        float[] normal = normal(p0, p1, p2);
        if (normal == null && p3 != p2) normal = normal(p0, p2, p3);
        if (normal == null) return; // zero-area polygon, nothing to draw
        if (!seen.add(faceKey(p0, p1, p2, p3, t0, t1, t2, t3))) return; // exact duplicate

        Face f = new Face();
        f.p = new float[][]{p0.clone(), p1.clone(), p2.clone(), p3.clone()};
        f.t = new float[][]{t0.clone(), t1.clone(), t2.clone(), t3.clone()};
        f.tri = p3 == p2;
        f.n = normal;
        out.add(f);
    }

    /** Builds the cubes and fills smooth per-vertex normals. */
    private static List<RawYsmModel.RawCube> finish(List<Face> faces) {
        int nf = faces.size();
        // corner -> group of corners that share the same position
        int[] gid = new int[nf * 4];
        Map<PosKey, Integer> groups = new HashMap<>(nf * 2);
        for (int f = 0; f < nf; f++) {
            Face face = faces.get(f);
            int corners = face.tri ? 3 : 4;
            for (int c = 0; c < 4; c++) {
                if (c >= corners) { gid[f * 4 + c] = -1; continue; }
                PosKey key = PosKey.of(face.p[c]);
                Integer g = groups.get(key);
                if (g == null) { g = groups.size(); groups.put(key, g); }
                gid[f * 4 + c] = g;
            }
        }
        int groupCount = groups.size();
        int[] start = new int[groupCount + 1];
        for (int i = 0; i < gid.length; i++) if (gid[i] >= 0) start[gid[i] + 1]++;
        for (int g = 0; g < groupCount; g++) start[g + 1] += start[g];
        int[] fill = start.clone();
        int[] entries = new int[start[groupCount]];
        for (int i = 0; i < gid.length; i++) if (gid[i] >= 0) entries[fill[gid[i]]++] = i;

        List<RawYsmModel.RawCube> out = new ArrayList<>(nf);
        for (int f = 0; f < nf; f++) {
            Face face = faces.get(f);
            int corners = face.tri ? 3 : 4;
            float[] vn = new float[12];
            for (int c = 0; c < corners; c++) {
                int g = gid[f * 4 + c];
                float sx = 0f, sy = 0f, sz = 0f;
                for (int k = start[g]; k < start[g + 1]; k++) {
                    int e = entries[k];
                    Face o = faces.get(e >> 2);
                    float dot = o.n[0] * face.n[0] + o.n[1] * face.n[1] + o.n[2] * face.n[2];
                    if (dot < CREASE_COS) continue; // hard edge
                    float w = cornerAngle(o, e & 3);
                    sx += o.n[0] * w;
                    sy += o.n[1] * w;
                    sz += o.n[2] * w;
                }
                float len = (float) Math.sqrt(sx * sx + sy * sy + sz * sz);
                if (len < MIN_NORMAL_LEN) { sx = face.n[0]; sy = face.n[1]; sz = face.n[2]; len = 1f; }
                vn[c * 3] = sx / len;
                vn[c * 3 + 1] = sy / len;
                vn[c * 3 + 2] = sz / len;
            }
            if (face.tri) { // 4th vertex repeats the 3rd
                vn[9] = vn[6];
                vn[10] = vn[7];
                vn[11] = vn[8];
            }

            RawYsmModel.RawFace rf = new RawYsmModel.RawFace();
            rf.normal = face.n;
            rf.vertexNormals = vn;
            rf.positions = face.p;
            rf.u = new float[]{face.t[0][0], face.t[1][0], face.t[2][0], face.t[3][0]};
            rf.v = new float[]{face.t[0][1], face.t[1][1], face.t[2][1], face.t[3][1]};

            RawYsmModel.RawCube cube = new RawYsmModel.RawCube();
            cube.faces.add(rf);
            out.add(cube);
        }
        return out;
    }

    /** Interior angle of the face at the given corner (radians, never 0 so degenerate corners still count a bit). */
    private static float cornerAngle(Face f, int c) {
        int n = f.tri ? 3 : 4;
        float[] o = f.p[c], a = f.p[(c + n - 1) % n], b = f.p[(c + 1) % n];
        float ax = a[0] - o[0], ay = a[1] - o[1], az = a[2] - o[2];
        float bx = b[0] - o[0], by = b[1] - o[1], bz = b[2] - o[2];
        float la = (float) Math.sqrt(ax * ax + ay * ay + az * az);
        float lb = (float) Math.sqrt(bx * bx + by * by + bz * bz);
        if (la < MIN_NORMAL_LEN || lb < MIN_NORMAL_LEN) return 1e-4f;
        float cos = (ax * bx + ay * by + az * bz) / (la * lb);
        cos = Math.max(-1f, Math.min(1f, cos));
        return Math.max(1e-4f, (float) Math.acos(cos));
    }

    /** Order-preserving, rotation-invariant 64 bit key of a face (vertices + UVs, exact float bits). */
    private static long faceKey(float[] p0, float[] p1, float[] p2, float[] p3,
                                float[] t0, float[] t1, float[] t2, float[] t3) {
        boolean tri = p3 == p2;
        int n = tri ? 3 : 4;
        long[] h = {vertexHash(p0, t0), vertexHash(p1, t1), vertexHash(p2, t2), vertexHash(p3, t3)};
        int start = 0;
        for (int i = 1; i < n; i++) if (h[i] < h[start]) start = i;
        long key = 0x9E3779B97F4A7C15L ^ n;
        for (int i = 0; i < n; i++) key = mix(key ^ h[(start + i) % n]);
        return key;
    }

    private static long vertexHash(float[] p, float[] t) {
        long k = 0x243F6A8885A308D3L;
        k = mix(k ^ Float.floatToIntBits(p[0]));
        k = mix(k ^ Float.floatToIntBits(p[1]));
        k = mix(k ^ Float.floatToIntBits(p[2]));
        k = mix(k ^ Float.floatToIntBits(t[0]));
        k = mix(k ^ Float.floatToIntBits(t[1]));
        return k;
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** (b-a)x(c-a), normalized; {@code null} for degenerate triangles. */
    private static float[] normal(float[] a, float[] b, float[] c) {
        float ax = b[0] - a[0], ay = b[1] - a[1], az = b[2] - a[2];
        float bx = c[0] - a[0], by = c[1] - a[1], bz = c[2] - a[2];
        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;
        float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len < MIN_NORMAL_LEN) return null;
        return new float[]{nx / len, ny / len, nz / len};
    }
}
