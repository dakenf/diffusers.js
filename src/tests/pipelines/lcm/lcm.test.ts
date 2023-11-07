import { setModelCacheDir } from '../../../hub/node'
import { DiffusionPipeline } from '../../../pipelines/DiffusionPipeline'

describe('LCM', () => {
  setModelCacheDir(__dirname +  '/../../examples/react/public/models')
  it ('should encode prompt', async () => {
    const pipe = await DiffusionPipeline.fromPretrained('aislamov/lcm-dreamshaper-fp16')

    const prompt = await pipe.getPromptEmbeds('An astronaut riding a green horse', 'car')
  })
})
