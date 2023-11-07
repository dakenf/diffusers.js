import { Tensor } from '@xenova/transformers'

export function betasForAlphaBar (
  numDiffusionTimesteps: number,
  maxBeta = 0.999,
  alphaTransformType: 'exp'|'cosine' = 'cosine',
) {
  function alphaBar (timeStep: number) {
    if (alphaTransformType === 'cosine') {
      return Math.cos((timeStep + 0.008) / 1.008 * Math.PI / 2) ** 2
    } else if (alphaTransformType === 'exp') {
      return Math.exp(timeStep * -12)
    }

    throw new Error('Unsupported alphaTransformType: ' + alphaTransformType)
  }

  const betas = []
  for (let i = 0; i < numDiffusionTimesteps; i++) {
    const t1 = i / numDiffusionTimesteps
    const t2 = (i + 1) / numDiffusionTimesteps
    betas.push(Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta))
  }
  return new Tensor(betas)
}
