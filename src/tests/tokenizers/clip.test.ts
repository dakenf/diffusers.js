import { CLIPTokenizer } from '../../tokenizers/CLIPTokenizer'
import { Tensor } from '@xenova/transformers'

const downloadOpts = {
  progress_callback: () => {},
}

describe('CLIP Tokenizer', () => {
  it ('should encode SD prompt', async () => {
    const tokenizer = await CLIPTokenizer.from_pretrained('aislamov/stable-diffusion-2-1-base-onnx', {...downloadOpts, subdir: 'tokenizer' })
    const result = tokenizer('An astronaut riding a horse', { return_tensor: false })
    expect(result.input_ids).toEqual([49406, 550, 18376, 6765, 320, 4558, 49407])

    const inputIds = result.input_ids
    const tensor = new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length])
  })
})
