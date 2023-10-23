import * as ORT from 'onnxruntime-node'
import { StableDiffusionXLPipeline } from '../../../pipelines/StableDiffusionXLPipeline'
import * as fs from 'fs'
import { CLIPTokenizer } from '../../../tokenizers/CLIPTokenizer'

// https://github.com/microsoft/onnxruntime/issues/16622
const originalImplementation = Array.isArray;
// @ts-ignore
Array.isArray = (type) => {
  if (type?.constructor?.name === "Float32Array" || type?.constructor?.name === "BigInt64Array") {
    return true;
  }

  return originalImplementation(type);
};

// @ts-ignore
const InferenceSession = ORT.default.InferenceSession

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

describe('SDXL', () => {
  it ('should encode prompt', async () => {
    const modelDir = 'public/models/aislamov/sdxl-base-fp32'
    const options = { executionProviders: ['cpu'] }
    const unet = await InferenceSession.create(`${modelDir}/unet/model.onnx`, options)
    const textEncoder = await InferenceSession.create(`${modelDir}/text_encoder/model.onnx`, options)
    const textEncoder2 = await InferenceSession.create(`${modelDir}/text_encoder_2/model.onnx`, options)
    const vae = await InferenceSession.create(`${modelDir}/vae_encoder/model.onnx`, options)

    const schedulerConfig = JSON.parse(fs.readFileSync(`${modelDir}/scheduler/scheduler_config.json`, 'utf-8'))
    const scheduler = StableDiffusionXLPipeline.createScheduler(schedulerConfig)

    const tokenizer = createTokenizer(`${modelDir}/tokenizer`)
    const tokenizer2 = createTokenizer(`${modelDir}/tokenizer_2`)

    const sd = new StableDiffusionXLPipeline(unet, vae, vae, textEncoder, textEncoder2, tokenizer, tokenizer2, scheduler)

    const prompt = await sd.getPromptEmbeds('An astronaut riding a green horse', 'car')

    // time
    const timeEmbeds = sd.getTimeEmbeds(1024, 1024)
    expect([...timeEmbeds.data]).toEqual([1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0, 1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0])
    expect([...timeEmbeds.dims]).toEqual([2, 6])

    const expectedPrompt = JSON.parse(fs.readFileSync('src/tests/pipelines/sdxl/prompt_embeds.json', 'utf-8'))
    expect([...prompt.hiddenStates.dims]).toEqual(expectedPrompt.dims)

    for (let idx = 0; idx < prompt.hiddenStates.data.length; idx++) {
      const val = prompt.hiddenStates.data[idx]
      const exp = expectedPrompt.data[idx]
      expect(val).toBeCloseTo(exp, 3)
    }
    // expect([...prompt.hiddenStates.data]).toEqual(expectedPrompt.data)

    const expectedTextEmbeds = JSON.parse(fs.readFileSync('src/tests/pipelines/sdxl/text_embeds.json', 'utf-8'))
    expect([...prompt.textEmbeds.dims]).toEqual(expectedTextEmbeds.dims)
    // expect([...prompt.textEmbeds.data]).toEqual(expectedTextEmbeds.data)

  })
})
