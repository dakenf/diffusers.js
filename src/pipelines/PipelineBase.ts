import { Session } from '@/backends'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { SchedulerBase } from '@/schedulers/SchedulerBase'
import { Tensor } from '@xenova/transformers'
import {cat, randomNormalTensor} from '@/util/Tensor'

export class PipelineBase {
  public unet: Session
  public vaeDecoder: Session
  public vaeEncoder: Session
  public textEncoder: Session
  public tokenizer: CLIPTokenizer
  public scheduler: SchedulerBase
  public vaeScaleFactor: number

  /**
   * Tokenizes and encodes the input prompt. Before encoding, we verify if it is necessary
   * to break the prompt into chunks due to the Tokenizer model limit (which is usually 77)
   * by getting the maximum length of the input prompt and comparing it with the Tokenizer
   * model max length. If the prompt exceeds the Tokenizer model limit, then it is
   * necessary to break the prompt into chunks, otherwise, it is not necessary.
   * 
   * @param prompt Input prompt.
   * @param highestTokenLength Highest token length between prompt and negative prompt or Tokenizer model max length.
   * @returns Tensor containing the prompt embeddings.
   */
  async encodePrompt (prompt: string, highestTokenLength: number): Promise<Tensor> {
    let tokens, encoded, inputIds;
    const TokenMaxLength = this.tokenizer.model_max_length // Tokenizer model max length of tokens including the <START> and <END> tokens

    if(highestTokenLength > TokenMaxLength) { // Prompt exceeds tokenizer model max length, therefore we need to use chunks
      let embeddingsTensorArray: Tensor[] = [] // Will contain all of the prompt token embedding chunks
      const userTokenMaxLength = TokenMaxLength - 2 // Max length of tokens minus the <START> and <END> tokens

      tokens = this.tokenizer(
        prompt,
        {
          return_tensor: false,
          padding: false,
          max_length: TokenMaxLength,
          return_tensor_dtype: 'int32',
        },
      )

      inputIds = tokens.input_ids // Tokenized prompt
      const START_token = inputIds.shift() // Remove <START> token
      const END_token = inputIds.pop() // Remove <END> token

      for(let i = 0; i < highestTokenLength; i += userTokenMaxLength) {
        let tokenChunk = inputIds.slice(i, i + userTokenMaxLength)
        
        for(let j = tokenChunk.length; j < userTokenMaxLength; j++) { // Pad chunk to userTokenMaxLength if necessary. Use the <END> token to pad.
          tokenChunk.push(END_token)
        }

        tokenChunk.unshift(START_token) // Add <START> token to each chunk
        tokenChunk.push(END_token) // Add <END> token to each chunk

        encoded = await this.textEncoder.run({ input_ids: new Tensor('int32', Int32Array.from(tokenChunk.flat()), [1, tokenChunk.length]) })
        embeddingsTensorArray.push(encoded.last_hidden_state)
      }

      return cat(embeddingsTensorArray, 1)
    }
    else { // Prompt that does not exceed tokenizer max length. Padding is used.
      tokens = this.tokenizer(
        prompt,
        {
          return_tensor: false,
          padding: true,
          max_length: TokenMaxLength,
          return_tensor_dtype: 'int32',
        },
      )

      inputIds = tokens.input_ids // Tokenized prompt
      encoded = await this.textEncoder.run({ input_ids: new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length]) })
      return encoded.last_hidden_state
    }
  }

  /**
   * Returns the prompt and negative prompt text embeddings.
   * 
   * @param prompt Input prompt.
   * @param negativePrompt Input negative prompt.
   * @returns Tensor containing the prompt and negative prompt embeddings.
   */
  async getPromptEmbeds (prompt: string, negativePrompt: string | undefined) {
    // We check which has more tokens between the prompt and negative prompt
    const promptTokens = this.tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: false,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const negPromptTokens = this.tokenizer(
      negativePrompt,
      {
        return_tensor: false,
        padding: false,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const promptTokensLength = promptTokens.input_ids.length // Number of tokens in prompt including the <START> and <END> tokens
    const negPromptTokensLength = negPromptTokens.input_ids.length // Number of tokens in negative prompt including the <START> and <END> tokens
    const highestTokenLength = Math.max(promptTokensLength, negPromptTokensLength)

    const promptEmbeds = await this.encodePrompt(prompt, highestTokenLength)
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '', highestTokenLength)

    return cat([negativePromptEmbeds, promptEmbeds])
  }

  prepareLatents (batchSize: number, numChannels: number, height: number, width: number, seed = '') {
    const latentShape = [
      batchSize,
      numChannels,
      Math.floor(width / this.vaeScaleFactor),
      height / this.vaeScaleFactor,
    ]

    return randomNormalTensor(latentShape, undefined, undefined, 'float32', seed)
  }

  async makeImages (latents: Tensor) {
    latents = latents.div(this.vaeDecoder.config.scaling_factor || 0.18215)

    const decoded = await this.vaeDecoder.run(
      { latent_sample: latents },
    )

    const images = decoded.sample
      .div(2)
      .add(0.5)
      .clipByValue(0, 1)
    // .mul(255)
    // .round()
    // .clipByValue(0, 255)
    // .transpose(0, 2, 3, 1)
    return [images]
  }

  async release () {
    await this.unet?.release()
    await this.vaeDecoder?.release()
    await this.vaeEncoder?.release()
    await this.textEncoder?.release()
  }
}
