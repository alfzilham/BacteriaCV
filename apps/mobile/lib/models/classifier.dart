import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'area_resize.dart';

/// Level keyakinan, dipakai di seluruh UI.
///
/// AMBANG INI WAJIB IDENTIK dengan aplikasi web dan dengan test di sana.
/// Nilainya 0,8 dan 0,6. Mengubahnya berarti aplikasi ini dan aplikasi
/// web memberi keputusan berbeda untuk model yang sama, jadi ambangnya
/// dikunci di sini dan diuji oleh `test/confidence_level_test.dart`.
enum ConfidenceLevel {
  /// confidence >= 0,8
  high('high'),

  /// confidence >= 0,6 dan < 0,8
  medium('medium'),

  /// confidence < 0,6
  low('low');

  const ConfidenceLevel(this.wireName);

  /// Nama yang dipakai untuk ditampilkan dan untuk diuji.
  final String wireName;
}

/// Batas level. Nilai float ditulis eksplisit supaya tidak ada
/// pembulatan diam-diam yang berbeda antara dua aplikasi.
const double highThreshold = 0.8;
const double mediumThreshold = 0.6;

/// Mengubah confidence menjadi level.
///
/// Fungsi ini sengaja terpisah dari model supaya ambangnya bisa diuji
/// tanpa model ONNX dan tanpa perangkat.
ConfidenceLevel levelForConfidence(double confidence) {
  if (confidence >= highThreshold) {
    return ConfidenceLevel.high;
  }
  if (confidence >= mediumThreshold) {
    return ConfidenceLevel.medium;
  }
  return ConfidenceLevel.low;
}

/// Batas ukuran berkas, 20 MB, sama seperti versi web.
///
/// Ditegakkan SEBELUM decode. Menolaknya lewat UI saja tidak cukup,
/// karena berkas besar sudah memakan memori sebelum ditolak.
const int maxFileBytes = 20 * 1024 * 1024;

/// Dilempar kalau berkas lebih besar dari [maxFileBytes].
class FileTooLargeException implements Exception {
  const FileTooLargeException(this.bytes);

  /// Ukuran berkas yang ditolak, dalam byte.
  final int bytes;

  @override
  String toString() =>
      'File is ${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB. '
      'The limit is ${maxFileBytes ~/ (1024 * 1024)} MB.';
}

/// Hasil klasifikasi satu citra.
class Classification {
  const Classification({
    required this.shapeLabel,
    required this.gramLabel,
    required this.shapeLevel,
    required this.gramLevel,
    required this.shapeConfidence,
    required this.gramConfidence,
    required this.elapsedMs,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// "cocci" atau "bacilli".
  final String shapeLabel;

  /// "positive" atau "negative".
  final String gramLabel;

  /// Level keyakinan bentuk sel.
  final ConfidenceLevel shapeLevel;

  /// Level keyakinan status Gram.
  final ConfidenceLevel gramLevel;

  /// Confidence bentuk sel dari softmax head A.
  ///
  /// TIDAK ditampilkan di UI. Model int8 meleset sampai 0,542656
  /// terhadap acuan float32 meskipun F1 tetap 1,000000, jadi angkanya tidak
  /// terkalibrasi dan memamerkannya sebagai persen akan menipu.
  final double shapeConfidence;

  /// Confidence status Gram dari sigmoid head B indeks 1.
  ///
  /// Sama seperti [shapeConfidence], tidak ditampilkan sebagai persen.
  final double gramConfidence;

  /// Waktu proses total, decode sampai inferensi selesai.
  final int elapsedMs;

  /// Lebar citra sumber, sebelum resize.
  final int imageWidth;

  /// Tinggi citra sumber, sebelum resize.
  final int imageHeight;
}

/// Menjalankan pipeline yang sudah divalidasi: decode TIFF dengan paket
/// `image`, resize INTER_AREA, lalu inferensi int8 lewat XNNPACK.
///
/// Pipeline ini tidak diubah dari yang sudah diuji. Yang dihapus hanya
/// pengukuran: tidak ada mode sweep, tidak ada pengulangan warm, tidak
/// ada catatan timing per tahap yang tidak dipakai user.
///
/// Ada satu jalur inferensi saja. Kalau nanti ada kebutuhan modality
/// lain, itu warrants PR sendiri, bukan jalur kedua di dalam kelas ini.
class BacteriaClassifier {
  BacteriaClassifier._(this._session);

  static const String _modelAsset = 'assets/spike/model_int8.onnx';
  static const int _imageSize = 224;
  static const List<String> _shapeLabels = <String>['cocci', 'bacilli'];
  static const List<String> _gramLabels = <String>['positive', 'negative'];

  final OrtSession _session;
  static BacteriaClassifier? _instance;

  /// Mengambil instance Tunggal, membuat sesi bila belum ada.
  ///
  /// Sesi ONNX dibuat sekali lalu dipakai ulang. Membuat sesi ulang
  /// untuk tiap citra akan menambah ratusan milidetik cold start yang
  /// perlu, dan pengukuran Task V menunjukkan cold total 379 ms
  /// sedangkan warm 72 ms.
  static Future<BacteriaClassifier> instance() async {
    final existing = _instance;
    if (existing != null) {
      return existing;
    }
    final session = await OnnxRuntime().createSessionFromAsset(
      _modelAsset,
      options: OrtSessionOptions(
        intraOpNumThreads: 4,
        interOpNumThreads: 1,
        // Hanya XNNPACK. NNAPI diukur 1.471 sampai 2.011 ms warm untuk
        // model int8 di perangkat ini, jauh lebih lambat dari XNNPACK
        // 72 ms, dan tidak dipakai.
        providers: const <OrtProvider>[OrtProvider.XNNPACK],
      ),
    );
    final created = BacteriaClassifier._(session);
    _instance = created;
    return created;
  }

  /// Melepaskan sesi. Dipakai test dan saat aplikasi ditutup.
  static Future<void> dispose() async {
    await _instance?._session.close();
    _instance = null;
  }

  /// Menjalankan pipeline penuh untuk satu berkas.
  ///
  /// Lempar [FileTooLargeException] kalau berkas melebihi 20 MB, dan
  /// lempar [StateError] kalau TIFF tidak bisa didecode.
  Future<Classification> classify(String filePath) async {
    final watch = Stopwatch()..start();

    // 1. Batas ukuran, sebelum berkas dibaca ke memori dan sebelum decode.
    final file = File(filePath);
    final length = await file.length();
    checkSize(length);

    // 2. Decode. Platform codec Android tidak punya decoder TIFF dan
    //    menolak dengan "Invalid image data", jadi paket `image` yang
    //    dipakai. Jalur ini sudah diukur 134 ms warm.
    final bytes = Uint8List.fromList(await file.readAsBytes());
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('The image could not be decoded.');
    }

    // 3. Resize. area_resize.dart dipakai apa adanya, tidak ditulis
    //    ulang: itu satu-satunya bagian pipeline yang sudah diuji
    //    presisi terhadap cv2 di dua skala.
    final resized = resizeArea(
      decoded.buffer.asUint8List(),
      decoded.width,
      decoded.height,
      _imageSize,
    );

    // 4. Inferensi. Masukan uint8 NHWC 224x224x3; normalisasi ada di
    //    dalam graf, bukan di sini.
    final input = await OrtValue.fromList(
      resized,
      [1, _imageSize, _imageSize, 3],
    );
    final outputs = await _session.run(<String, OrtValue>{'image_u8': input});
    await input.dispose();

    final headA = outputs['head_a_logits'];
    final headB = outputs['head_b_logits'];
    if (headA == null || headB == null) {
      throw StateError('Model output not found.');
    }
    final logitA =
        (await headA.asFlattenedList())
            .map((v) => (v as num).toDouble())
            .toList();
    final logitB =
        (await headB.asFlattenedList())
            .map((v) => (v as num).toDouble())
            .toList();
    for (final value in outputs.values) {
      await value.dispose();
    }
    watch.stop();

    // 5. Label. Aturan ini sama dengan yang dipakai di spike dan
    //    dihitung ulang dari logit, bukan diambil dari sesi.
    final probabilities = _softmax(logitA);
    final shapeIndex = probabilities.indexOf(probabilities.reduce(math.max));
    final gramProbability = 1 / (1 + math.exp(-logitB[1]));
    final gramIndex = gramProbability > 0.5 ? 1 : 0;
    final shapeConfidence = probabilities[shapeIndex];
    final gramConfidence = gramIndex == 1 ? gramProbability : 1 - gramProbability;

    return Classification(
      shapeLabel: _shapeLabels[shapeIndex],
      gramLabel: _gramLabels[gramIndex],
      shapeLevel: levelForConfidence(shapeConfidence),
      gramLevel: levelForConfidence(gramConfidence),
      shapeConfidence: shapeConfidence,
      gramConfidence: gramConfidence,
      elapsedMs: watch.elapsedMilliseconds,
      imageWidth: decoded.width,
      imageHeight: decoded.height,
    );
  }

  /// Menegakkan batas ukuran tanpa perlu model, supaya bisa diuji.
  static void checkSize(int bytes) {
    if (bytes > maxFileBytes) {
      throw FileTooLargeException(bytes);
    }
  }

  List<double> _softmax(List<double> values) {
    final maxValue = values.reduce(math.max);
    final exponentials = values.map((v) => math.exp(v - maxValue)).toList();
    final total = exponentials.fold<double>(0, (a, b) => a + b);
    return exponentials.map((v) => v / total).toList();
  }
}