/// Resize citra ke ukuran tetap dengan INTER_AREA, implementasi pure Dart.
///
/// Kenapa ada di Dart dan bukan di dalam graf model:
///
/// BacteriaCV me-resize ke 224 x 224 dengan `cv2.INTER_AREA` sebelum
/// menormalisasi. Rasio 2048/224 = 9,142857 dan 1532/224 = 6,839285 bukan
/// bilangan bulat, jadi INTER_AREA TIDAK bisa diekspresikan sebagai
/// operator ONNX maupun TFLite yang builtin. Keduanya hanya punya resize
/// bilinear dan nearest-neighbor. Memaksakan operator itu akan diam-diam
/// mengubah preprocessing yang sedang diukur.
///
/// Kenapa normalisasi TETAP di dalam graf: (/255 - mean) / std dihitung
/// di float32 oleh runtime ONNX, satu-satunya implementasi. Kalau
/// dihitung di Dart, pembulatan float32 di dua bahasa bisa berbeda dan
/// angka keluarannya tidak lagi bisa dibandingkan dengan PyTorch.
///
/// ATURAN JENDELA - INI BUKAN BOX AVERAGE BIASA
/// ---------------------------------------------
/// `INTER_AREA` OpenCV memberi bobot PERSIANGAN pada piksel tepi jendela,
/// bukan membagi rata semua piksel di dalam kotak integer.
///
/// Untuk piksel tujuan d pada sumbu sepanjang n, dengan s = n / out:
///
///   a = d * s                 batas bawah jendela pecahan
///   b = (d + 1) * s           batas atas jendela pecahan
///   i0 = floor(a)             piksel pertama yang ikut
///   i1 = ceil(b) - 1          piksel terakhir yang ikut
///   bobot[0]  = ceil(a) - a   bagian piksel kiri yang masuk jendela
///   bobot[1:-1] = 1           piksel di dalam
///   bobot[-1] = b - floor(b)  bagian piksel kanan yang masuk jendela
///
/// Jendela 2-D memakai hasil kali luar bobot sumbu-Y dan bobot sumbu-X.
///
/// Dua versi spike yang lebih dulu salah, keduanya terdeteksi oleh
/// test/area_resize_test.dart terhadap acuan cv2:
///
///   1. box average dengan floor dan ceil tanpa bobot pecahan - jendela
///      melebar dan bergeser sampai satu piksel penuh, selisih sampai 18
///      satuan pada 58 persen byte;
///   2. box average dengan pembulatan integer - lebih dekat, tapi tidak
///      pernah tepat, karena OpenCV menghitung overlap secara pecahan.
///
/// Dengan aturan di atas, selisih yang tersisa adalah pembulatan murni:
/// maksimal 1 satuan pada sekitar 0,2 persen piksel pada citra DIBaS
/// 768 x 574. Itu yang diuji test, dengan batas kedua: kalau lebih dari
/// 1 persen byte yang meleset, itu bukan pembulatan, itu algoritma
/// yang salah dan test harus gagal.
library;

import 'dart:typed_data';

/// Jendela satu sumbu: indeks awal, indeks akhir inklusif, dan bobot.
class _AxisWindow {
  const _AxisWindow(this.start, this.end, this.weights);

  final int start;
  final int end;

  /// Bobot pecahan per piksel, sepanjang `end - start + 1`.
  final List<double> weights;

  int get length => end - start + 1;
}

/// Hitung jendela dan bobot INTER_AREA untuk satu sumbu.
///
/// [length] adalah panjang sumber, [output] panjang tujuan.
List<_AxisWindow> _axisWindows(int length, int output) {
  final scale = length / output;
  final windows = <_AxisWindow>[];

  for (var d = 0; d < output; d++) {
    final a = d * scale;
    final b = (d + 1) * scale;

    var start = a.floor();
    var end = b.ceil() - 1;
    if (start < 0) start = 0;
    if (end > length - 1) end = length - 1;
    if (end < start) end = start;

    final count = end - start + 1;
    final weights = List<double>.filled(count, 1.0);
    if (count > 1) {
      final left = a.ceil() - a;
      final right = b - b.floor();
      // Batas yang jatuh tepat di tepi piksel tidak diberi bobot pecahan:
      // bukan 0, karena 0 akan menghilangkan piksel itu dari pembilang
      // sementara tetap dihitung di penyebut.
      if (left > 1e-12) weights[0] = left;
      if (right > 1e-12) weights[count - 1] = right;
    }

    windows.add(_AxisWindow(start, end, weights));
  }

  return windows;
}

/// Resize RGB uint8 HWC ke [output] x [output] dengan INTER_AREA.
///
/// [src] adalah buffer RGB uint8 tersusun HWC (height, width, 3).
/// Mengembalikan buffer RGB uint8 HWC berukuran `output * output * 3`.
Uint8List resizeArea(Uint8List src, int srcWidth, int srcHeight, int output) {
  assert(src.length >= srcWidth * srcHeight * 3);
  final dst = Uint8List(output * output * 3);

  final windowsY = _axisWindows(srcHeight, output);
  final windowsX = _axisWindows(srcWidth, output);

  for (var dy = 0; dy < output; dy++) {
    final windowY = windowsY[dy];
    for (var dx = 0; dx < output; dx++) {
      final windowX = windowsX[dx];

      var sumR = 0.0;
      var sumG = 0.0;
      var sumB = 0.0;
      var weightSum = 0.0;

      for (var yi = 0; yi < windowY.length; yi++) {
        final wy = windowY.weights[yi];
        final rowOffset = (windowY.start + yi) * srcWidth;
        for (var xi = 0; xi < windowX.length; xi++) {
          final w = wy * windowX.weights[xi];
          final i = (rowOffset + windowX.start + xi) * 3;
          sumR += src[i] * w;
          sumG += src[i + 1] * w;
          sumB += src[i + 2] * w;
          weightSum += w;
        }
      }

      final o = (dy * output + dx) * 3;
      dst[o] = _roundHalfUp(sumR / weightSum);
      dst[o + 1] = _roundHalfUp(sumG / weightSum);
      dst[o + 2] = _roundHalfUp(sumB / weightSum);
    }
  }

  return dst;
}

/// Pembulatan setengah ke atas, mengikuti aritmetika OpenCV
/// `(sum + area / 2) / area`.
int _roundHalfUp(double value) => (value + 0.5).floor();