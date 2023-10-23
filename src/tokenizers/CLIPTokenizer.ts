import { PreTrainedTokenizer, PretrainedOptions, Tensor } from '@xenova/transformers'
import { getModelJSON, getModelTextFile } from '@/hub'

interface TokenizerOptions {
  text_pair?: null|string
  padding?: boolean
  truncation?: boolean
  max_length?: number|null
  return_tensor?: boolean
  return_tensor_dtype?: string
}

interface ClipPreTrainedOptions extends PretrainedOptions {
  subdir?: string
  revision?: string
}

export class CLIPTokenizer extends PreTrainedTokenizer {
  private readonly bos_token_id?: number
  private readonly eos_token_id?: number
  constructor (tokenizerJSON: unknown, tokenizerConfig: unknown) {
    super(tokenizerJSON, tokenizerConfig)
    this.added_tokens_regex = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/gui
    // this.pad_token_id = 0

    const bos_token = this.getToken(tokenizerConfig, 'bos_token')
    if (bos_token) {
      this.bos_token_id = this.model.tokens_to_ids.get(bos_token)
    }

    const eos_token = this.getToken(tokenizerConfig, 'eos_token')
    if (eos_token) {
      this.eos_token_id = this.model.tokens_to_ids.get(eos_token)
    }
  }

  _call (
    // Required positional arguments
    text: string,

    // Optional keyword arguments
    {
      text_pair = null,
      // add_special_tokens = true, // TODO
      padding = false,
      truncation = false,
      max_length = null,
      return_tensor = true, // Different to HF
      return_tensor_dtype = 'int64',
    }: TokenizerOptions = {},
  ): { input_ids: number[]|number[][]|Tensor, attention_mask: Tensor[]|Tensor } {
    let tokens: number[][]

    if (Array.isArray(text)) {
      if (text.length === 0) {
        throw Error('text array must be non-empty')
      }

      if (text_pair !== null) {
        if (!Array.isArray(text_pair)) {
          throw Error('text_pair must also be an array')
        } else if (text.length !== text_pair.length) {
          throw Error('text and text_pair must have the same length')
        }

        tokens = text.map(
          (t, i) => this.encode(t, text_pair[i]),
        )
      } else {
        tokens = text.map(x => this.encode(x))
      }
    } else {
      if (text === null) {
        throw Error('text may not be null')
      }

      if (Array.isArray(text_pair)) {
        throw Error('When specifying `text_pair`, since `text` is a string, `text_pair` must also be a string (i.e., not an array).')
      }

      // For single input, we just wrap in an array, and then unwrap later.
      tokens = [this.encode(text, text_pair)]
    }
    // At this point, tokens is batched: [batch_size, tokens]
    // However, array may be jagged. So, we pad to max_length

    const maxLengthOfBatch = Math.max(...tokens.map(x => x.length))

    // If null, we calculate max length from sequences
    if (max_length === null) {
      max_length = maxLengthOfBatch
    }

    // Ensure it is less than model max length
    max_length = Math.min(max_length, this.model_max_length)

    if (this.bos_token_id) {
      // Add the BOS token
      tokens = tokens.map(x => [this.bos_token_id!].concat(x))
    }

    if (this.eos_token_id) {
      // Add the EOS token
      tokens = tokens.map(x => x.concat([this.eos_token_id!]))
    }

    /** @type {any[]|Tensor} */
    let attention_mask = []
    if (padding || truncation) {
      // Perform padding and/or truncation
      for (let i = 0; i < tokens.length; ++i) {
        if (tokens[i].length === max_length) {
          attention_mask.push(new Array(tokens[i].length).fill(1))
          continue
        } else if (tokens[i].length > max_length) {
          // possibly truncate
          if (truncation) {
            tokens[i] = tokens[i].slice(0, max_length)
          }
          attention_mask.push(new Array(tokens[i].length).fill(1))
        } else { // t.length < max_length
          if (padding) {
            const diff = max_length - tokens[i].length

            if (this.padding_side === 'right') {
              attention_mask.push(
                (new Array(tokens[i].length).fill(1)).concat(new Array(diff).fill(0)),
              )
              tokens[i].push(...new Array(diff).fill(this.pad_token_id))
            } else { // left
              attention_mask.push(
                (new Array(diff).fill(0)).concat(new Array(tokens[i].length).fill(1)),
              )
              tokens[i].unshift(...new Array(diff).fill(this.pad_token_id))
            }
          } else {
            attention_mask.push(new Array(tokens[i].length).fill(1))
          }
        }
      }
    } else {
      attention_mask = tokens.map(x => new Array(x.length).fill(1))
    }

    if (return_tensor) {
      if (!(padding && truncation)) {
        // Not, guaranteed that all items have same length, so
        // we perform additional check

        if (tokens.some(x => x.length !== tokens[0].length)) {
          throw Error(
            'Unable to create tensor, you should probably activate truncation and/or padding ' +
            "with 'padding=true' and 'truncation=true' to have batched tensors with the same length.",
          )
        }
      }

      // Now we actually convert to tensor
      // NOTE: In the same way as the python library, we return a batched tensor, regardless of
      // whether we have a single input or multiple inputs.
      const dims = [tokens.length, tokens[0].length]

      if (return_tensor_dtype === 'int32') {
        // @ts-ignore
        tokens = new Tensor(return_tensor_dtype,
          Int32Array.from(tokens.flat()),
          dims,
        )

        // @ts-ignore
        attention_mask = new Tensor(
          return_tensor_dtype,
          Int32Array.from(attention_mask.flat()),
          dims,
        )
      } else {
        // @ts-ignore
        tokens = new Tensor(return_tensor_dtype,
          BigInt64Array.from(tokens.flat().map(BigInt)),
          dims,
        )

        // @ts-ignore
        attention_mask = new Tensor(
          return_tensor_dtype,
          BigInt64Array.from(attention_mask.flat().map(BigInt)),
          dims,
        )
      }
    } else {
      // If not returning a tensor, we match the input type
      if (!Array.isArray(text)) {
        // Input was not batched, so we unwrap
        // @ts-ignore
        tokens = tokens[0]
        attention_mask = attention_mask[0]
      }
    }

    // Finally, add attention mask, and possibly model-specific parameters
    let modelInputs = {
      input_ids: tokens,
      attention_mask,
    }

    // Optional post-processing
    modelInputs = this.prepare_model_inputs(modelInputs)

    return modelInputs
  }

  _encode_text (text: string|null): string[] | null {
    if (text === null) {
      return []
    }

    // Actual function which does encoding, for a single text
    // First, we take care of special tokens. Needed to avoid issues arising from
    // normalization and/or pretokenization (which may not preserve special tokens)
    const sections = [...text.matchAll(this.added_tokens_regex)].map(x => x[0])

    return sections.map(x => {
      if (this.added_tokens.includes(x)) {
        // Ignore added tokens
        return x
      } else {
        if (this.remove_space === true) {
          x = x.trim().split(/\s+/).join(' ')
        }

        if (this.normalizer !== null) {
          x = this.normalizer(x)
        }

        const sectionTokens = (this.pre_tokenizer !== null) ? this.pre_tokenizer(x) : [x]

        return this.model(sectionTokens)
      }
    }).flat()
  }

  static async from_pretrained (pretrained_model_name_or_path: string, options: ClipPreTrainedOptions = { subdir: 'tokenizer' }) {
    const [vocab, merges, tokenizerConfig] = await Promise.all([
      getModelJSON(pretrained_model_name_or_path, `${options.subdir}/vocab.json`, true, { revision: options.revision }),
      // getModelJSON(pretrained_model_name_or_path, `${options.subdir}/special_tokens_map.json`, true, options),
      getModelTextFile(pretrained_model_name_or_path, `${options.subdir}/merges.txt`, true, { revision: options.revision }),
      getModelJSON(pretrained_model_name_or_path, `${options.subdir}/tokenizer_config.json`, true, { revision: options.revision }),
    ])
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
}
