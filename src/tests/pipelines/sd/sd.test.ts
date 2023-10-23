import fs from 'fs'
import { CLIPTokenizer } from '../../../tokenizers/CLIPTokenizer'
import { Session } from '../../../backends'
import { StableDiffusionPipeline } from '../../../pipelines/StableDiffusionPipeline'


function createTokenizer (tokenizerPath: string) {
  const vocab = JSON.parse(fs.readFileSync(`${tokenizerPath}/vocab.json`, 'utf-8'))
  const tokens = JSON.parse(fs.readFileSync(`${tokenizerPath}/special_tokens_map.json`, 'utf-8'))
  const merges = fs.readFileSync(`${tokenizerPath}/merges.txt`, 'utf-8')
  const tokenizerConfig = JSON.parse(fs.readFileSync(`${tokenizerPath}/tokenizer_config.json`, 'utf-8'))
  const tokenizerJSON = {
    normalizer: {
      type: 'Lowercase',
    },
    pre_tokenizer: {
      type: 'WhitespaceSplit',
    },
    post_processor: {
      type: 'ByteLevel',
    },
    decoder: {
      type: 'ByteLevel',
    },
    model: {
      type: 'BPE',
      vocab,
      use_regex: true,
      end_of_word_suffix: '</w>',
      merges: merges!.split('\n').slice(1, 49152 - 256 - 2 + 1),
    },
    added_tokens: [],
  }

  return new CLIPTokenizer(tokenizerJSON, tokenizerConfig)
}

describe('SD', () => {
  it ('should encode prompt', async () => {
    const modelDir = 'public/models/aislamov/sd2_1base-fp16'
    const options = { executionProviders: ['cpu'] }
    const unet = await Session.create(`${modelDir}/unet/model.onnx`)
    const textEncoder = await Session.create(`${modelDir}/text_encoder/model.onnx`)
    const vae = await Session.create(`${modelDir}/vae_encoder/model.onnx`)

    const schedulerConfig = JSON.parse(fs.readFileSync(`${modelDir}/scheduler/scheduler_config.json`, 'utf-8'))
    const scheduler = StableDiffusionPipeline.createScheduler(schedulerConfig)

    const tokenizer = createTokenizer(`${modelDir}/tokenizer`)

    const sd = new StableDiffusionPipeline(unet, vae, vae, textEncoder, tokenizer, scheduler)

    const prompt = await sd.getPromptEmbeds('An astronaut riding a green horse', 'car')
  })
})
