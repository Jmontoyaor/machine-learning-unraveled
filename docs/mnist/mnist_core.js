/*
 * MNIST inference engine for the browser (no external libraries).
 * - Preprocessing: same canonical pipeline as mnist_app/preprocessing.py
 *   (black background = 0, white stroke = 1, crop, fit 20x20 keeping aspect,
 *   center by mass in 28x28).
 * - Forward pass of the two reference Keras models exported by export_web.py.
 * Works in browsers (window.MnistCore) and in Node (module.exports) for tests.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MnistCore = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var MIN_CONTRAST = 20, INK_THRESHOLD = 24, MIN_INK_PIXELS = 3, BORDER_MARGIN = 2;
  var EMPTY = "empty";

  function decodeBase64(text) {
    if (typeof Buffer !== "undefined") {
      var buf = Buffer.from(text, "base64");
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.length);
    }
    var bin = atob(text), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function decodeFloat32(text) {
    var bytes = decodeBase64(text), copy = new Uint8Array(bytes.length);
    copy.set(bytes);
    return new Float32Array(copy.buffer);
  }

  /* ---------------------------------------------------------- preprocessing */
  function median(values) {
    var s = Array.prototype.slice.call(values).sort(function (a, b) { return a - b; });
    var m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function otsuThreshold(pixels) {
    var hist = new Array(256).fill(0), total = pixels.length, sum = 0, i;
    for (i = 0; i < total; i++) hist[pixels[i]]++;
    for (i = 0; i < 256; i++) sum += i * hist[i];
    var sumB = 0, wB = 0, best = 0, threshold = 0;
    for (i = 0; i < 256; i++) {
      wB += hist[i];
      if (!wB) continue;
      var wF = total - wB;
      if (!wF) break;
      sumB += i * hist[i];
      var mB = sumB / wB, mF = (sum - sumB) / wF, between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; threshold = i; }
    }
    return threshold;
  }

  /* Area-average resampling: exact for downscaling, smooth for upscaling. */
  function resizeArea(src, sw, sh, dw, dh) {
    var out = new Float32Array(dw * dh), sx = sw / dw, sy = sh / dh;
    for (var y = 0; y < dh; y++) {
      var y0 = y * sy, y1 = y0 + sy;
      for (var x = 0; x < dw; x++) {
        var x0 = x * sx, x1 = x0 + sx, acc = 0, area = 0;
        for (var yy = Math.floor(y0); yy < Math.min(sh, Math.ceil(y1)); yy++) {
          var wy = Math.min(y1, yy + 1) - Math.max(y0, yy);
          if (wy <= 0) continue;
          for (var xx = Math.floor(x0); xx < Math.min(sw, Math.ceil(x1)); xx++) {
            var wx = Math.min(x1, xx + 1) - Math.max(x0, xx);
            if (wx <= 0) continue;
            acc += src[yy * sw + xx] * wx * wy;
            area += wx * wy;
          }
        }
        out[y * dw + x] = area ? acc / area : 0;
      }
    }
    return out;
  }

  function shiftToCenterOfMass(img) {
    var total = 0, sr = 0, sc = 0, r, c;
    for (r = 0; r < 28; r++) for (c = 0; c < 28; c++) {
      var v = img[r * 28 + c]; total += v; sr += r * v; sc += c * v;
    }
    if (total <= 0) return img;
    var dy = Math.round(14 - sr / total), dx = Math.round(14 - sc / total);
    var out = new Float32Array(784);
    for (r = 0; r < 28; r++) for (c = 0; c < 28; c++) {
      var nr = r + dy, nc = c + dx;
      if (nr >= 0 && nr < 28 && nc >= 0 && nc < 28) out[nr * 28 + nc] = img[r * 28 + c];
    }
    return out;
  }

  /*
   * gray: Uint8 grayscale pixels (row-major), width x height.
   * options.invert: null/undefined = automatic, true = invert, false = keep.
   * options.clean: true removes paper/shadow noise with an Otsu threshold (camera).
   * Returns { image: Float32Array(784) in [0,1], info } or throws Error("empty").
   */
  function preprocess(gray, width, height, options) {
    options = options || {};
    var n = width * height, i, min = 255, max = 0;
    for (i = 0; i < n; i++) { if (gray[i] < min) min = gray[i]; if (gray[i] > max) max = gray[i]; }
    if (max - min < MIN_CONTRAST) throw new Error(EMPTY);

    var border = [];
    for (var x = 0; x < width; x++) border.push(gray[x], gray[(height - 1) * width + x]);
    for (var y = 0; y < height; y++) border.push(gray[y * width], gray[y * width + width - 1]);
    var detectedInvert = median(border) > 127;
    var invert = options.invert == null ? detectedInvert : !!options.invert;

    var ink = new Uint8Array(n);
    for (i = 0; i < n; i++) ink[i] = invert ? 255 - gray[i] : gray[i];

    var cleanThreshold = null;
    if (options.clean) {
      cleanThreshold = Math.max(INK_THRESHOLD, otsuThreshold(ink));
      var span = Math.max(1, 255 - cleanThreshold);
      for (i = 0; i < n; i++) ink[i] = ink[i] <= cleanThreshold ? 0 : Math.min(255, Math.round((ink[i] - cleanThreshold) * 255 / span));
    }

    var useMargin = Math.min(width, height) > 4 * BORDER_MARGIN;
    var x0 = width, y0 = height, x1 = -1, y1 = -1, count = 0;
    for (y = 0; y < height; y++) for (x = 0; x < width; x++) {
      if (useMargin && (x < BORDER_MARGIN || y < BORDER_MARGIN || x >= width - BORDER_MARGIN || y >= height - BORDER_MARGIN)) continue;
      if (ink[y * width + x] > INK_THRESHOLD) {
        count++;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
    if (count < MIN_INK_PIXELS) throw new Error(EMPTY);

    var cw = x1 - x0 + 1, ch = y1 - y0 + 1, cropped = new Uint8Array(cw * ch);
    for (y = 0; y < ch; y++) for (x = 0; x < cw; x++) cropped[y * cw + x] = ink[(y0 + y) * width + x0 + x];

    var scale = 20 / Math.max(cw, ch);
    var nw = Math.max(1, Math.round(cw * scale)), nh = Math.max(1, Math.round(ch * scale));
    var resized = resizeArea(cropped, cw, ch, nw, nh);
    var boxed = new Float32Array(784), ox = (28 - nw) >> 1, oy = (28 - nh) >> 1;
    for (y = 0; y < nh; y++) for (x = 0; x < nw; x++)
      boxed[(oy + y) * 28 + ox + x] = Math.min(1, Math.max(0, Math.round(resized[y * nw + x]) / 255)); // normalize once
    var finalImage = shiftToCenterOfMass(boxed);

    return {
      image: finalImage,
      info: {
        detectedInvert: detectedInvert, inverted: invert, cleanThreshold: cleanThreshold,
        steps: {
          grayscale: { data: gray, width: width, height: height, scale: 255 },
          polarity: { data: ink, width: width, height: height, scale: 255 },
          cropped: { data: cropped, width: cw, height: ch, scale: 255 },
          boxed: { data: boxed, width: 28, height: 28, scale: 1 },
          final: { data: finalImage, width: 28, height: 28, scale: 1 }
        }
      }
    };
  }

  /* ---------------------------------------------------------- inference */
  function loadModel(spec) {
    return spec.layers.map(function (layer) {
      var copy = Object.assign({}, layer);
      if (layer.kernel) { copy.kernelData = decodeFloat32(layer.kernel); copy.biasData = decodeFloat32(layer.bias); }
      return copy;
    });
  }

  function dense(input, layer, activation) {
    var inSize = layer.kernel_shape[0], out = layer.kernel_shape[1], k = layer.kernelData;
    var y = new Float32Array(out);
    for (var j = 0; j < out; j++) y[j] = layer.biasData[j];
    for (var i = 0; i < inSize; i++) {
      var xi = input[i];
      if (xi === 0) continue;
      var row = i * out;
      for (j = 0; j < out; j++) y[j] += xi * k[row + j];
    }
    if (activation === "relu") for (j = 0; j < out; j++) if (y[j] < 0) y[j] = 0;
    return y;
  }

  /* Valid 3x3 convolution, channels-last (HWC), followed by ReLU. */
  function conv2d(input, h, w, cin, layer) {
    var kh = layer.kernel_shape[0], kw = layer.kernel_shape[1], cout = layer.kernel_shape[3];
    var oh = h - kh + 1, ow = w - kw + 1, k = layer.kernelData, b = layer.biasData;
    var out = new Float32Array(oh * ow * cout);
    for (var y = 0; y < oh; y++) for (var x = 0; x < ow; x++) {
      var base = (y * ow + x) * cout;
      for (var co = 0; co < cout; co++) out[base + co] = b[co];
      for (var u = 0; u < kh; u++) for (var v = 0; v < kw; v++) {
        var inBase = ((y + u) * w + (x + v)) * cin, kBase = (u * kw + v) * cin;
        for (var ci = 0; ci < cin; ci++) {
          var xv = input[inBase + ci];
          if (xv === 0) continue;
          var kOff = (kBase + ci) * cout;
          for (co = 0; co < cout; co++) out[base + co] += xv * k[kOff + co];
        }
      }
      for (co = 0; co < cout; co++) if (out[base + co] < 0) out[base + co] = 0;
    }
    return { data: out, h: oh, w: ow, c: cout };
  }

  function softmax(z) {
    var m = -Infinity, s = 0, i, out = new Float32Array(z.length);
    for (i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    for (i = 0; i < z.length; i++) { out[i] = Math.exp(z[i] - m); s += out[i]; }
    for (i = 0; i < z.length; i++) out[i] /= s;
    return out;
  }

  /* Runs any sequence of Dense / Conv2D / Flatten layers; Keras flattens HWC in the same order. */
  function predict(layers, image28) {
    var x = image28, shape = [28, 28, 1], activations = {};
    for (var i = 0; i < layers.length; i++) {
      var layer = layers[i];
      if (layer.type === "Conv2D") {
        if (layer.activation !== "relu") throw new Error("Unsupported Conv2D activation: " + layer.activation);
        var r = conv2d(x, shape[0], shape[1], shape[2], layer);
        x = r.data; shape = [r.h, r.w, r.c];
        activations[layer.name] = r;
      } else if (layer.type === "Flatten") {
        shape = [x.length];
      } else if (layer.type === "Dense") {
        if (layer.activation === "softmax") x = softmax(dense(x, layer, "linear"));
        else if (layer.activation === "relu" || layer.activation === "linear") x = dense(x, layer, layer.activation);
        else throw new Error("Unsupported Dense activation: " + layer.activation);
      } else if (layer.type !== "InputLayer") {
        throw new Error("Unsupported layer: " + layer.type);
      }
    }
    return { probabilities: x, activations: activations };
  }

  return {
    EMPTY: EMPTY, decodeBase64: decodeBase64, decodeFloat32: decodeFloat32,
    otsuThreshold: otsuThreshold, preprocess: preprocess, loadModel: loadModel, predict: predict
  };
});
