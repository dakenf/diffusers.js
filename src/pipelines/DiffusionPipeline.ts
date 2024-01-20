import { PretrainedOptions } from '@/pipelines/common'
import { GetModelFileOptions } from '@/hub/common'
import { getModelJSON } from '@/hub'
import { StableDiffusionPipeline } from '@/pipelines/StableDiffusionPipeline'
import { StableDiffusionControlNetPipeline } from './StableDiffusionControlNetPipeline'
import { StableDiffusionXLPipeline } from '@/pipelines/StableDiffusionXLPipeline'
import { LatentConsistencyModelPipeline } from '@/pipelines/LatentConsistencyModelPipeline'
import { LCMScheduler } from '@/schedulers/LCMScheduler'

export class DiffusionPipeline {
  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    const index = await getModelJSON(modelRepoOrPath, 'model_index.json', true, opts)
    let pipe: StableDiffusionXLPipeline
    switch (index['_class_name']) {
      case 'StableDiffusionPipeline':
      case 'OnnxStableDiffusionPipeline':
        if (typeof index.controlnet !== 'undefined') {
          return StableDiffusionControlNetPipeline.fromPretrained(modelRepoOrPath, options)
        }
        // temp hack to identify the SD Turbo model
        if (index.scheduler[1] === 'EulerDiscreteScheduler') {
          return SDTurboPipeline.fromPretrained(modelRepoOrPath, options)
        }
        return StableDiffusionPipeline.fromPretrained(modelRepoOrPath, options)
      case 'StableDiffusionXLPipeline':
      case 'ORTStableDiffusionXLPipeline':
        return StableDiffusionXLPipeline.fromPretrained(modelRepoOrPath, options)
      case 'LCMStableDiffusionXLPipeline':
        pipe = await StableDiffusionXLPipeline.fromPretrained(modelRepoOrPath, options)
        // @ts-ignore
        pipe.scheduler = new LCMScheduler(pipe.scheduler.config)
        return pipe
      case 'LatentConsistencyModelPipeline':
        return LatentConsistencyModelPipeline.fromPretrained(modelRepoOrPath, options)
      default:
        throw new Error(`Unknown pipeline type ${index['_class_name']}`)
    }
  }
}
