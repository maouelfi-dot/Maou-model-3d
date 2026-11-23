package com.micaftic.morpher.resource;

import com.micaftic.morpher.resource.pojo.RawYsmModel;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PolyMeshBakerTest {

    private static int[][] tri(int a, int b, int c) {
        return new int[][]{{a, a, a}, {b, b, b}, {c, c, c}};
    }

    @Test
    void triangleBecomesSingleFaceCubeWithDegenerateFourthVertex() {
        float[][] pos = {{0, 0, 0}, {16, 0, 0}, {0, 16, 0}};
        float[][] uv = {{0.25f, 0.75f}, {0.5f, 0.75f}, {0.25f, 0.5f}};
        List<RawYsmModel.RawCube> cubes = PolyMeshBaker.bake(pos, uv, new int[][][]{tri(0, 1, 2)}, true, 64, 64);
        assertEquals(1, cubes.size());
        RawYsmModel.RawFace f = cubes.get(0).faces.get(0);
        assertEquals(f.positions[2][0], f.positions[3][0]);
        assertEquals(f.positions[2][1], f.positions[3][1]);
        // x is mirrored (-x/16), vertex order reversed
        assertEquals(0f, f.positions[0][0], 1e-6f);
        assertEquals(-1f, f.positions[1][0], 1e-6f);
        // CCW outward normal of the source triangle is +Z and stays +Z after the mirror
        assertTrue(f.normal[2] > 0.99f);
    }

    @Test
    void normalizedUvsFlipV() {
        float[][] pos = {{0, 0, 0}, {16, 0, 0}, {0, 16, 0}};
        float[][] uv = {{0.1f, 0.2f}, {0.3f, 0.2f}, {0.1f, 0.4f}};
        RawYsmModel.RawFace f = PolyMeshBaker.bake(pos, uv, new int[][][]{tri(0, 1, 2)}, true, 64, 64)
                .get(0).faces.get(0);
        // vertex order is reversed: face vertex 0 is source vertex 2
        assertEquals(0.1f, f.u[0], 1e-6f);
        assertEquals(1f - 0.4f, f.v[0], 1e-6f);
    }

    @Test
    void pixelUvsAreDividedByTextureSize() {
        float[][] pos = {{0, 0, 0}, {16, 0, 0}, {0, 16, 0}};
        float[][] uv = {{8, 16}, {16, 16}, {8, 32}};
        RawYsmModel.RawFace f = PolyMeshBaker.bake(pos, uv, new int[][][]{tri(0, 1, 2)}, false, 64, 64)
                .get(0).faces.get(0);
        assertEquals(8f / 64f, f.u[0], 1e-6f);
        assertEquals(32f / 64f, f.v[0], 1e-6f);
    }

    @Test
    void quadStaysQuadAndNgonIsFanTriangulated() {
        float[][] pos = {{0, 0, 0}, {16, 0, 0}, {16, 16, 0}, {0, 16, 0}, {-8, 8, 0}};
        float[][] uv = {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5f, 0.5f}};
        int[][] quad = {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
        int[][] pent = {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
        assertEquals(1, PolyMeshBaker.bake(pos, uv, new int[][][]{quad}, true, 16, 16).size());
        assertEquals(3, PolyMeshBaker.bake(pos, uv, new int[][][]{pent}, true, 16, 16).size());
    }

    @Test
    void degenerateAndInvalidPolysAreSkipped() {
        float[][] pos = {{1, 1, 1}, {1, 1, 1}, {2, 2, 2}};
        float[][] uv = {{0, 0}};
        int[][][] polys = {tri(0, 1, 2), {{9, 0, 0}, {0, 0, 0}, {1, 0, 0}}, {{0, 0, 0}, {1, 0, 0}}};
        assertTrue(PolyMeshBaker.bake(pos, uv, polys, true, 16, 16).isEmpty());
    }

    @Test
    void geometryParserReadsPolyMeshBones() {
        String json = "{\"format_version\":\"1.21.0\",\"minecraft:geometry\":[{\"description\":{\"identifier\":\"geometry.t\","
                + "\"texture_width\":64,\"texture_height\":64},\"bones\":[{\"name\":\"Head\",\"pivot\":[0,24,0],"
                + "\"poly_mesh\":{\"normalized_uvs\":true,\"positions\":[[0,0,0],[16,0,0],[0,16,0]],"
                + "\"normals\":[[0,0,0],[0,0,0],[0,0,0]],\"uvs\":[[0,0],[1,0],[0,1]],"
                + "\"polys\":[[[0,0,0],[1,1,1],[2,2,2]]]}}]}]}";
        RawYsmModel.RawGeometry geo = YsmGeometryParsing.parseGeometry(json.getBytes(StandardCharsets.UTF_8), 1);
        assertEquals(1, geo.bones.size());
        assertEquals(1, geo.bones.get(0).cubes.size());
    }

    @Test
    void gentleFoldIsSmoothedAcrossSharedEdge() {
        float[][] pos = {{0, 0, 0}, {16, 0, 0}, {0, 16, 0}, {16, 16, 6}};
        List<RawYsmModel.RawCube> cubes = PolyMeshBaker.bake(pos, new float[0][], new int[][][]{tri(0, 1, 2), tri(1, 3, 2)}, true, 64, 64);
        assertEquals(2, cubes.size());
        for (RawYsmModel.RawCube cube : cubes) {
            RawYsmModel.RawFace f = cube.faces.get(0);
            assertTrue(f.vertexNormals != null && f.vertexNormals.length == 12);
            boolean anySmoothed = false;
            for (int v = 0; v < 3; v++) {
                float dot = f.vertexNormals[v * 3] * f.normal[0] + f.vertexNormals[v * 3 + 1] * f.normal[1]
                        + f.vertexNormals[v * 3 + 2] * f.normal[2];
                if (dot < 0.9999f) anySmoothed = true;
            }
            assertTrue(anySmoothed, "vertices on the shared edge must differ from the flat face normal");
        }
    }

    @Test
    void hardEdgeStaysCrisp() {
        float[][] pos = {{0, 0, 0}, {16, 0, 0}, {0, 16, 0}, {0, 0, 16}};
        List<RawYsmModel.RawCube> cubes = PolyMeshBaker.bake(pos, new float[0][], new int[][][]{tri(0, 1, 2), tri(0, 2, 3)}, true, 64, 64);
        for (RawYsmModel.RawCube cube : cubes) {
            RawYsmModel.RawFace f = cube.faces.get(0);
            for (int v = 0; v < 3; v++) {
                assertEquals(f.normal[0], f.vertexNormals[v * 3], 1e-5f);
                assertEquals(f.normal[1], f.vertexNormals[v * 3 + 1], 1e-5f);
                assertEquals(f.normal[2], f.vertexNormals[v * 3 + 2], 1e-5f);
            }
        }
    }
}
