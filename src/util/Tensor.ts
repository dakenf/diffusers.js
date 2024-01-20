import { Tensor as ONNXTensor } from 'onnxruntime-common'
import { Tensor } from '@xenova/transformers'
import seedrandom from 'seedrandom'

Tensor.prototype.reverse = function () {
  return new Tensor(this.type, this.data.reverse(), this.dims.slice())
}

Tensor.prototype.sub = function (value: Tensor) {
  return this.clone().sub_(value)
}

Tensor.prototype.sub_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}

Tensor.prototype.add = function (value: Tensor) {
  return this.clone().add_(value)
}

Tensor.prototype.add_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}

Tensor.prototype.cumprod = function (dim: number) {
  return this.clone().cumprod_(dim)
}

Tensor.prototype.cumprod_ = function (dim: number) {
  const newDims = this.dims.slice()
  // const newStrides = this.strides.slice();
  if (dim === undefined) {
    dim = this.dims.length - 1
  }
  if (dim < 0 || dim >= this.dims.length) {
    throw new Error(`Invalid dimension: ${dim}`)
  }
  const size = newDims[dim]
  for (let i = 1; i < size; ++i) {
    for (let j = 0; j < this.data.length / size; ++j) {
      const index = j * size + i
      this.data[index] *= this.data[index - 1]
    }
  }
  // newDims[dim] = 1;
  // newStrides[dim] = 0;
  return this
}

Tensor.prototype.mul = function (value: Tensor|number) {
  return this.clone().mul_(value)
}

Tensor.prototype.mul_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}

Tensor.prototype.div = function (value: Tensor|number) {
  return this.clone().div_(value)
}

Tensor.prototype.div_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value.data[i]
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}

Tensor.prototype.pow = function (value: Tensor|number) {
  return this.clone().pow_(value)
}

Tensor.prototype.pow_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value)
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes')
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value.data[i])
    }
  } else {
    throw new Error('Invalid argument')
  }
  return this
}

Tensor.prototype.round = function () {
  return this.clone().round_()
}

Tensor.prototype.round_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.round(this.data[i])
  }
  return this
}

Tensor.prototype.tile = function (reps: number|number[]) {
  return this.clone().tile_(reps)
}

Tensor.prototype.tile_ = function (reps: number|number[]) {
  if (typeof reps === 'number') {
    reps = [reps]
  }
  if (reps.length < this.dims.length) {
    throw new Error('Invalid number of repetitions')
  }
  const newDims = []
  const newStrides = []
  for (let i = 0; i < this.dims.length; ++i) {
    newDims.push(this.dims[i] * reps[i])
    newStrides.push(this.strides[i])
  }
  const newData = new this.data.constructor(newDims.reduce((a, b) => a * b))
  for (let i = 0; i < newData.length; ++i) {
    let index = 0
    for (let j = 0; j < this.dims.length; ++j) {
      index += Math.floor(i / newDims[j]) * this.strides[j]
    }
    newData[i] = this.data[index]
  }
  return new Tensor(this.type, newData, newDims)
}

Tensor.prototype.clipByValue = function (min: number, max: number) {
  return this.clone().clipByValue_(min, max)
}

Tensor.prototype.clipByValue_ = function (min: number, max: number) {
  if (max < min) {
    throw new Error('Invalid arguments')
  }
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.min(Math.max(this.data[i], min), max)
  }
  return this
}

Tensor.prototype.exp = function () {
  return this.clone().exp_()
}

Tensor.prototype.exp_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.exp(this.data[i])
  }
  return this
}

Tensor.prototype.sin = function () {
  return this.clone().sin_()
}

Tensor.prototype.sin_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.sin(this.data[i])
  }
  return this
}

Tensor.prototype.cos = function () {
  return this.clone().cos_()
}

Tensor.prototype.cos_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.cos(this.data[i])
  }
  return this
}

Tensor.prototype.sqrt = function () {
  return this.clone().sqrt_()
}

Tensor.prototype.sqrt_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.sqrt(this.data[i])
  }
  return this
}

export function interp (
  x: Tensor,
  xp: Tensor,
  fp: Tensor,
) {
  if (xp.dims.length !== 1) {
    throw new Error('xp must be 1 dimensional')
  }
  if (fp.dims.length !== 1) {
    throw new Error('fp must be 1 dimensional')
  }
  if (xp.dims[0] !== fp.dims[0]) {
    throw new Error('xp and fp must have the same length')
  }
  if (x.dims.length !== 1) {
    throw new Error('x must be 1 dimensional')
  }
  const newDims = x.dims.slice()
  // @ts-ignore
  const newData = new x.data.constructor(newDims.reduce((a, b) => a * b))
  const left = fp.data[0]
  const right = fp.data[fp.data.length - 1]
  for (let i = 0; i < newData.length; ++i) {
    const index = xp.data.findIndex((v) => v > x.data[i])
    if (index === -1) {
      newData[i] = right
    } else if (index === 0) {
      newData[i] = left
    } else {
      const x1 = xp.data[index - 1]
      const x2 = xp.data[index]
      const y1 = fp.data[index - 1]
      const y2 = fp.data[index]
      newData[i] = ((x.data[i] - x1) * (y2 - y1)) / (x2 - x1) + y1
    }
  }
  return new Tensor(x.type, newData, newDims)
}

Tensor.prototype.location = 'cpu'

export function range (start: number, end: number, step = 1, type = 'float32') {
  const data = []
  for (let i = start; i < end; i += step) {
    data.push(i)
  }
  return new Tensor(type, data, [data.length])
}

export function linspace (start: number, end: number, num: number, type = 'float32') {
  const arr = []
  const step = (end - start) / (num - 1)
  for (let i = 0; i < num; i++) {
    arr.push(start + step * i)
  }
  return new Tensor(type, arr, [num])
}

function randomNormal (rng: seedrandom.PRNG) {
  let u = 0; let v = 0

  while (u === 0) u = rng()
  while (v === 0) v = rng()
  const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  return num
}

export function scalarTensor (num: number, type = 'float32') {
  return new Tensor(type, new Float32Array([num]), [1])
}

export function randomNormalTensor (shape: number[], mean = 0, std = 1, type = 'float32', seed: string = '') {
  const data = []
  const rng = seed !== '' ? seedrandom(seed) : seedrandom()
  for (let i = 0; i < shape.reduce((a, b) => a * b); i++) {
    data.push(randomNormal(rng) * std + mean)
  }
  return new Tensor(type, data, shape)
}

/**
 * Concatenates an array of tensors along the 0th dimension.
 *
 * @param {any} tensors The array of tensors to concatenate.
 * @returns {Tensor} The concatenated tensor.
 */
export function cat (tensors: Tensor[], axis: number = 0) {
  if (tensors.length === 0) {
    throw new Error('No tensors provided.')
  }

  // Handle negative axis by converting it to its positive counterpart
  if (axis < 0) {
    axis = tensors[0].dims.length + axis
  }

  const tensorType = tensors[0].type
  const tensorShape = [...tensors[0].dims]

  // Ensure all tensors have the same shape except for the concatenation axis
  for (const t of tensors) {
    for (let i = 0; i < tensorShape.length; i++) {
      if (i !== axis && tensorShape[i] !== t.dims[i]) {
        throw new Error('Tensor dimensions must match for concatenation, except along the specified axis.')
      }
    }
  }

  // Calculate the size of the concatenated tensor along the specified axis
  tensorShape[axis] = tensors.reduce((sum, t) => sum + t.dims[axis], 0)

  // Calculate total size to allocate
  const total = tensorShape.reduce((product, size) => product * size, 1)

  // Create output tensor of same type as the first tensor
  const data = new tensors[0].data.constructor(total)

  let offset = 0
  for (const t of tensors) {
    const copySize = t.data.length / t.dims[axis] // size of each slice along the axis
    for (let i = 0; i < t.dims[axis]; i++) {
      const sourceStart = i * copySize
      const sourceEnd = sourceStart + copySize
      data.set(t.data.slice(sourceStart, sourceEnd), offset)
      offset += copySize
    }
  }

  return new Tensor(tensorType, data, tensorShape)
}

export function replaceTensors (modelRunResult: Record<string, ONNXTensor>): Record<string, Tensor> {
  // Convert ONNX Tensors with our custom Tensor class
  // to support additional functions
  const result: Record<string, Tensor> = {}
  for (const prop in modelRunResult) {
    if (modelRunResult[prop].dims) {
      // @ts-ignore
      result[prop] = new Tensor(
        // @ts-ignore
        modelRunResult[prop].type,
        // @ts-ignore
        modelRunResult[prop].data,
        // @ts-ignore
        modelRunResult[prop].dims,
      )
    }
  }
  return result
}
