import { Tensor as ONNXTensor } from 'onnxruntime-common'
import { Tensor } from '@xenova/transformers'

Tensor.prototype.reverse = function () {
  return new Tensor(this.type, this.data.reverse(), this.dims.slice());
}

Tensor.prototype.sub = function (value: Tensor) {
  return this.clone().sub_(value);
}

Tensor.prototype.sub_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
}

Tensor.prototype.add = function (value: Tensor) {
  return this.clone().add_(value);
}

Tensor.prototype.add_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot subtract tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
}

Tensor.prototype.cumprod = function (dim: number) {
  return this.clone().cumprod_(dim);
}

Tensor.prototype.cumprod_ = function (dim: number) {
  const newDims = this.dims.slice();
  //const newStrides = this.strides.slice();
  if (dim === undefined) {
    dim = this.dims.length - 1;
  }
  if (dim < 0 || dim >= this.dims.length) {
    throw new Error(`Invalid dimension: ${dim}`);
  }
  const size = newDims[dim];
  for (let i = 1; i < size; ++i) {
    for (let j = 0; j < this.data.length / size; ++j) {
      const index = j * size + i;
      this.data[index] *= this.data[index - 1];
    }
  }
  // newDims[dim] = 1;
  //newStrides[dim] = 0;
  return this;
}

Tensor.prototype.mul = function (value: Tensor|number) {
  return this.clone().mul_(value);
}

Tensor.prototype.mul_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
}

Tensor.prototype.div = function (value: Tensor|number) {
  return this.clone().div_(value);
}

Tensor.prototype.div_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value.data[i];
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
}

Tensor.prototype.pow = function (value: Tensor|number) {
  return this.clone().pow_(value);
}

Tensor.prototype.pow_ = function (value: Tensor|number) {
  if (typeof value === 'number') {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value);
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error('Cannot multiply tensors of different sizes');
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value.data[i]);
    }
  } else {
    throw new Error('Invalid argument');
  }
  return this;
}

Tensor.prototype.round = function () {
  return this.clone().round_();
}

Tensor.prototype.round_ = function () {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.round(this.data[i]);
  }
  return this;
}

Tensor.prototype.tile = function (reps: number|number[]) {
  return this.clone().tile_(reps);
}

Tensor.prototype.tile_ = function (reps: number|number[]) {
  if (typeof reps === 'number') {
    reps = [reps];
  }
  if (reps.length < this.dims.length) {
    throw new Error('Invalid number of repetitions');
  }
  const newDims = [];
  const newStrides = [];
  for (let i = 0; i < this.dims.length; ++i) {
    newDims.push(this.dims[i] * reps[i]);
    newStrides.push(this.strides[i]);
  }
  const newData = new this.data.constructor(newDims.reduce((a, b) => a * b));
  for (let i = 0; i < newData.length; ++i) {
    let index = 0;
    for (let j = 0; j < this.dims.length; ++j) {
      index += Math.floor(i / newDims[j]) * this.strides[j];
    }
    newData[i] = this.data[index];
  }
  return new Tensor(this.type, newData, newDims);
}

Tensor.prototype.clipByValue = function (min: number, max: number) {
  return this.clone().clipByValue_(min, max);
}

Tensor.prototype.clipByValue_ = function (min: number, max: number) {
  if (max < min) {
    throw new Error('Invalid arguments');
  }
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.min(Math.max(this.data[i], min), max);
  }
  return this;
}


export function range (start: number, end: number, step = 1, type = 'float32') {
  let data = [];
  for (let i = start; i < end; i += step) {
    data.push(i);
  }
  return new Tensor(type, data, [data.length]);
}

export function linspace(start: number, end: number, num: number, type = 'float32') {
  const arr = [];
  const step = (end - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    arr.push(start + step * i);
  }
  return new Tensor(type, arr, [num]);
}

function randomNormal() {
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
  return num;
}

export function randomNormalTensor(shape: number[], mean = 0, std = 1, type = 'float32') {
  let data = [];
  for (let i = 0; i < shape.reduce((a, b) => a * b); i++) {
    data.push(randomNormal() * std + mean);
  }
  return new Tensor(type, data, shape);
}

/**
 * Concatenates an array of tensors along the 0th dimension.
 *
 * @param {any} tensors The array of tensors to concatenate.
 * @returns {Tensor} The concatenated tensor.
 */
export function cat(tensors: Tensor[]) {
  if (tensors.length === 0) {
    return tensors[0];
  }
  // NOTE: tensors must be batched
  // NOTE: currently only supports dim=0
  // TODO: add support for dim != 0


  let tensorType = tensors[0].type;
  let tensorShape = [...tensors[0].dims];
  tensorShape[0] = tensors.length;

  // Calculate total size to allocate
  let total = 0;
  for (let t of tensors) {
    total += t.data.length;
  }

  if (tensorShape.length === 1) {
    // 1D tensors are concatenated into a 1D tensor
    tensorShape = [total];
  }

  // Create output tensor of same type as first
  let data = new tensors[0].data.constructor(total);

  let offset = 0;
  for (let t of tensors) {
    data.set(t.data, offset);
    offset += t.data.length;
  }

  return new Tensor(tensorType, data, tensorShape)
}

export function replaceTensors (modelRunResult: any) {
  // Convert ONNX Tensors with our custom Tensor class
  // to support additional functions
  for (let prop in modelRunResult) {
    // @ts-ignore
    if (modelRunResult[prop].dims) {
      modelRunResult[prop] = new Tensor(modelRunResult[prop].type, modelRunResult[prop].data, modelRunResult[prop].dims);
    }
  }
  return modelRunResult;
}
