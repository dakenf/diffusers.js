import path from 'path'
import { GetModelFileOptions, pathJoin } from '@/hub/common'
import fs from 'fs'
import progress from 'cli-progress'
import { downloadFile } from '@huggingface/hub'

let cacheDir = '.cache'

export function setModelCacheDir (dir: string) {
  cacheDir = dir
}

async function fileExists (filePath: string) {
  try {
    await fs.promises.access(filePath)
    return true
  } catch (error) {
    return false
  }
}

export function getCacheKey (modelRepoOrPath: string, fileName: string, revision: string) {
  const filePath = pathJoin(cacheDir, modelRepoOrPath, revision === 'main' ? '' : revision, fileName)
  return path.resolve(filePath)
}

async function writeResponseToFile (response: Response, displayName: string, outputPath: string): Promise<void> {
  const totalSize = parseInt(response.headers.get('content-length') || '0')
  const progressBar = new progress.SingleBar({
    format: `Downloading ${displayName} | {bar} | {percentage}% | {size}/{totalFormatted}`,
  }, progress.Presets.shades_classic)

  progressBar.start(totalSize, 0, { totalFormatted: formatSize(totalSize) })

  const writeStream = fs.createWriteStream(outputPath + '.tmp')
  const reader = response.body!.getReader()

  // Helper function to format size
  function formatSize (size: number): string {
    if (size < 1024) return `${size} Bytes`
    else if (size < 1048576) return `${(size / 1024).toFixed(2)} KB`
    else if (size < 1073741824) return `${(size / 1048576).toFixed(2)} MB`
    return `${(size / 1073741824).toFixed(2)} GB`
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }
    writeStream.write(value)
    progressBar.increment(value!.length)
    progressBar.update({ size: formatSize(progressBar.value) })
  }
  await new Promise(resolve => writeStream.end(resolve))
  await fs.promises.rename(outputPath + '.tmp', outputPath)
}

export async function getModelFile (modelRepoOrPath: string, fileName: string, fatal = true, options: GetModelFileOptions = {}) {
  const revision = options.revision || 'main'
  const cachePath = getCacheKey(modelRepoOrPath, fileName, revision)
  if (await fileExists(cachePath)) {
    if (options.returnText) {
      return fs.promises.readFile(cachePath, { encoding: 'utf-8' })
    }

    return cachePath
  }

  // download model to cache
  try {
    const response = await downloadFile({ repo: modelRepoOrPath, path: fileName, revision })
    const targetPath = path.dirname(cachePath)
    if (!await fileExists(targetPath)) {
      await fs.promises.mkdir(targetPath, { recursive: true })
    }
    await writeResponseToFile(response, fileName, cachePath)

    // await fs.writeFile(cachePath, response.body as unknown as Stream);
    if (options.returnText) {
      return fs.promises.readFile(cachePath, { encoding: 'utf-8' })
    }

    return cachePath
  } catch (e) {
    if (!fatal) {
      return null
    }
    throw e
  }
}

export default {
  getModelFile,
}
