import { downloadFile } from '@huggingface/hub'
import { DbCache } from './indexed-db'

let cacheDir = '.cache';

export function setModelCacheDir(dir: string) {
  cacheDir = dir;
}


function pathJoin(...parts: string[]) {
  // https://stackoverflow.com/a/55142565
  parts = parts.map((part, index) => {
    if (index) {
      part = part.replace(new RegExp('^/'), '');
    }
    if (index !== parts.length - 1) {
      part = part.replace(new RegExp('/$'), '');
    }
    return part;
  })
  return parts.filter(p => p !== '').join('/');
}


function getCacheKey (modelRepoOrPath: string, fileName: string, revision: string) {
  const filePath = pathJoin(cacheDir, modelRepoOrPath, revision === 'main' ? '' : revision, fileName);
  // if (isNode) {
  //   return path.resolve(filePath);
  // }

  return filePath;
}

// options compatible with @xenova/transformers
interface GetModelFileOptions {
  progress_callback?: (data: { status: string, file: string, progress: number }) => void
  revision?: string
}

export async function getModelFile (modelRepoOrPath: string, fileName: string, fatal = true, options: GetModelFileOptions = {}) {
  const revision = options.revision || 'main';
  const cachePath = getCacheKey(modelRepoOrPath, fileName, revision)
  // if (isNode) {
  //   if (await fileExists(cachePath)) {
  //     return cachePath;
  //   }
  //
  //   // download model to cache
  //   // const response = await downloadFile({ repo: modelRepoOrPath, path: fileName, revision })
  //   // const targetPath = path.dirname(cachePath);
  //   // if (!await fileExists(targetPath)) {
  //   //   await fs.promises.mkdir(targetPath, { recursive: true })
  //   // }
  //   // const writeStream = createWriteStream(cachePath);
  //   // await pipelineAsync(response.body as unknown as ReadStream, writeStream);
  //
  //   return cachePath;
  // }

  // now browser, first try browser cache
  const cache = new DbCache()
  await cache.init()
  const cachedData = await cache.retrieveFile(cachePath);
  if (cachedData) {
    return cachedData.file;
  }

  let response: Response|null|undefined
  // now local cache
  if (cacheDir) {
    response = await fetch(cachePath);
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
      throw new Error(`Error downloading ${fileName}`);
    }

    const buffer = await readResponseToBuffer(response);
    await cache.storeFile(buffer, cachePath);

    return buffer;
  } catch (e) {
    if (!fatal) {
      return null;
    }
    throw e;
  }
}

export async function getModelJSON(modelPath: string, fileName: string, fatal = true, options = {}) {
  let buffer = await getModelFile(modelPath, fileName, fatal, options);
  if (buffer === null) {
    // Return empty object
    return {}
  }

  let decoder = new TextDecoder('utf-8');
  let jsonData = decoder.decode(buffer);

  return JSON.parse(jsonData);
}

function readResponseToBuffer (response: Response): Promise<ArrayBuffer> {
  const contentLength = response.headers.get('content-length');
  if (!contentLength) {
    return response.arrayBuffer();
  }

  let buffer: ArrayBuffer;
  const contentLengthNum = parseInt(contentLength, 10);

  if (contentLengthNum > 2 * 1024 * 1024 * 1024) {
    // @ts-ignore
    const memory = new WebAssembly.Memory({ initial: Math.ceil(contentLengthNum / 65536), index: 'i64' })
    buffer = memory.buffer
  } else {
    buffer = new ArrayBuffer(contentLengthNum)
  }

  let offset = 0
  return new Promise((resolve, reject) => {
    const reader = response.body!.getReader();

    function pump(): Promise<void> {
      return reader.read().then(({ done, value }) => {
        if (done) {
          return resolve(buffer);
        }
        const chunk = new Uint8Array(buffer, offset, value.byteLength)
        chunk.set(new Uint8Array(value))
        offset += value.byteLength
        return pump()
      });
    }

    pump().catch(reject)
  })
}
