/// <reference types="vite/client" />

// WGSL `?raw` imports are Vite-specific — the `?raw` suffix tells Vite to
// return the file contents as a string instead of processing it.
declare module "*.wgsl?raw" {
  const content: string;
  export default content;
}

// `?worker` imports return a constructor for a Web Worker.
declare module "*.ts?worker" {
  const workerConstructor: {
    new (options?: { name?: string }): Worker;
  };
  export default workerConstructor;
}
