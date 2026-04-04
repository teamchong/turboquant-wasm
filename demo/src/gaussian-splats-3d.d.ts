declare module "@mkkellogg/gaussian-splats-3d" {
  import type { Camera } from "three";

  export enum SceneRevealMode {
    Default = 0,
    Gradual = 1,
    Instant = 2,
  }

  export enum SceneFormat {
    Splat = 0,
    KSplat = 1,
    Ply = 2,
    Spz = 3,
  }

  export interface ViewerOptions {
    rootElement?: HTMLElement;
    selfDrivenMode?: boolean;
    sphericalHarmonicsDegree?: number;
    sceneRevealMode?: SceneRevealMode;
    gpuAcceleratedSort?: boolean;
    sharedMemoryForWorkers?: boolean;
    useBuiltInControls?: boolean;
    initialCameraPosition?: number[];
    initialCameraLookAt?: number[];
  }

  export interface AddSceneOptions {
    splatAlphaRemovalThreshold?: number;
    showLoadingUI?: boolean;
    progressiveLoad?: boolean;
    sphericalHarmonicsDegree?: number;
    format?: number;
  }

  export class Viewer {
    camera: Camera;
    constructor(options?: ViewerOptions);
    addSplatScene(path: string, options?: AddSceneOptions): Promise<void>;
    update(): void;
    render(): void;
    start(): void;
    dispose(): void;
  }
}
