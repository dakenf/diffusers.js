// import 'module-alias/register.js'
import nodeCache from '@/hub/node'
import { setCacheImpl } from '@/hub'
import { onnxruntimeBackend } from 'onnxruntime-node/dist/backend.js'
import * as ORT from '@aislamov/onnxruntime-web64'

export * from './pipelines/StableDiffusionPipeline'
export * from './pipelines/StableDiffusionXLPipeline'
export * from './pipelines/DiffusionPipeline'
export * from './pipelines/common'
export * from './hub'
export { setModelCacheDir } from '@/hub/browser'

// @ts-ignore
const ONNX = ORT.default ?? ORT
ONNX.registerBackend('cpu', onnxruntimeBackend, 1002)
setCacheImpl(nodeCache)
