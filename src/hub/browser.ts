import { downloadFile } from '@huggingface/hub'
import { DbCache } from '@/hub/indexed-db'
import { GetModelFileOptions, pathJoin } from '@/hub/common'
import { dispatchProgress, ProgressCallback, ProgressStatus } from '@/pipelines/common'

let cacheDir = ''
export function setModelCacheDir (dir: string) {
  cacheDir = dir
}

export function getCacheKey (modelRepoOrPath: string, fileName: string, revision: string) {
  return pathJoin(cacheDir, modelRepoOrPath, revision === 'main' ? '' : revision, fileName)
}

export async function getModelFile (modelRepoOrPath: string, fileName: string, fatal = true, options: GetModelFileOptions = {}) {
  const revision = options.revision || 'main'
  const cachePath = getCacheKey(modelRepoOrPath, fileName, revision)
  const cache = new DbCache()
  await cache.init()
  const cachedData = await cache.retrieveFile(cachePath, options.progressCallback, fileName)
  if (cachedData) {
    if (options.returnText) {
      const decoder = new TextDecoder('utf-8')
      return decoder.decode(cachedData.file)
    }

    return cachedData.file
  }

  let response: Response|null|undefined
  // now local cache
  if (cacheDir) {
    response = await fetch(cachePath)
    // create-react-app will return 200 with HTML for missing files
    if (!response || !response.body || response.status !== 200 || response.headers.get('content-type')?.startsWith('text/html')) {
      response = null
    }
  }

  try {
    // now try the hub
    if (!response) {
      response = await downloadFile({ repo: modelRepoOrPath, path: fileName, revision })
    }

    // read response
    if (!response || !response.body || response.status !== 200 || response.headers.get('content-type')?.startsWith('text/html')) {
      throw new Error(`Error downloading ${fileName}`)
    }

    const buffer = await readResponseToBuffer(response, options.progressCallback, fileName)
    await cache.storeFile(buffer, cachePath)
    if (options.returnText) {
      const decoder = new TextDecoder('utf-8')
      return decoder.decode(buffer)
    }

    return buffer
  } catch (e) {
    if (!fatal) {
      return null
    }
    throw e
  }
}

function readResponseToBuffer (response: Response, progressCallback: ProgressCallback, displayName: string): Promise<ArrayBuffer> {
  const contentLength = response.headers.get('content-length')
  if (!contentLength) {
    return response.arrayBuffer()
  }

  let buffer: ArrayBuffer
  const contentLengthNum = parseInt(contentLength, 10)

  if (contentLengthNum > 2 * 1024 * 1024 * 1024) {
    // @ts-ignore
    const memory = new WebAssembly.Memory({ initial: Math.ceil(contentLengthNum / 65536), index: 'i64' })
    buffer = memory.buffer
  } else {
    buffer = new ArrayBuffer(contentLengthNum)
  }

  let offset = 0
  return new Promise((resolve, reject) => {
    const reader = response.body!.getReader()

    async function pump (): Promise<void> {
      const { done, value } = await reader.read()
      if (done) {
        return resolve(buffer)
      }
      const chunk = new Uint8Array(buffer, offset, value.byteLength)
      chunk.set(new Uint8Array(value))
      offset += value.byteLength
      await dispatchProgress(progressCallback, {
        status: ProgressStatus.Downloading,
        downloadStatus: {
          file: displayName,
          size: contentLengthNum,
          downloaded: offset,
        }
      })
      return pump()
    }

    pump().catch(reject)
  })
}

export default {
  getModelFile,
}
