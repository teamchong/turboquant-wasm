/**
 * Dual GaussianSplats3D viewers with camera sync.
 */

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

export interface ViewerPair {
  left: InstanceType<typeof GaussianSplats3D.Viewer>;
  right: InstanceType<typeof GaussianSplats3D.Viewer>;
  startSync(): void;
  dispose(): void;
}

export function createViewerPair(
  leftEl: HTMLElement,
  rightEl: HTMLElement,
  shDegree: number = 2,
): ViewerPair {
  // Scene center: (5, -68, -60), extent: ~330 units
  const cameraOpts = {
    initialCameraPosition: [5, -68, 100],
    initialCameraLookAt: [5, -68, -60],
  };

  const left = new GaussianSplats3D.Viewer({
    ...cameraOpts,
    rootElement: leftEl,
    sphericalHarmonicsDegree: shDegree,
    sharedMemoryForWorkers: false,
  });

  const right = new GaussianSplats3D.Viewer({
    ...cameraOpts,
    rootElement: rightEl,
    sphericalHarmonicsDegree: shDegree,
    sharedMemoryForWorkers: false,
    useBuiltInControls: false,
  });

  let syncRunning = false;
  let rafId = 0;

  function syncCameras() {
    if (!syncRunning) return;
    rafId = requestAnimationFrame(syncCameras);

    const leftCam = left.camera;
    const rightCam = right.camera;
    if (leftCam && rightCam) {
      rightCam.position.copy(leftCam.position);
      rightCam.quaternion.copy(leftCam.quaternion);
      rightCam.projectionMatrix.copy(leftCam.projectionMatrix);
      rightCam.projectionMatrixInverse.copy(leftCam.projectionMatrixInverse);
    }
  }

  return {
    left,
    right,
    startSync() {
      syncRunning = true;
      syncCameras();
    },
    dispose() {
      syncRunning = false;
      cancelAnimationFrame(rafId);
    },
  };
}

export async function loadScene(
  viewer: InstanceType<typeof GaussianSplats3D.Viewer>,
  url: string,
  shDegree: number = 2,
  format?: number,
): Promise<void> {
  await viewer.addSplatScene(url, {
    splatAlphaRemovalThreshold: 5,
    sphericalHarmonicsDegree: shDegree,
    ...(format !== undefined ? { format } : {}),
  });
  viewer.start();
}
