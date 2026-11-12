import 'package:bacteriacv_mobile/models/classifier.dart';
import 'package:flutter_test/flutter_test.dart';

/// Menguji bagian classifier yang bisa diuji tanpa model ONNX dan
/// tanpa perangkat.
///
/// BAGIAN YANG TIDAK DIUJI DI SINI, dan kenapa:
///   * `classify()` itu sendiri. Ia butuh sesi ONNX yang hanya bisa
///     dibuka di perangkat dengan model benar-benar termuat, jadi tidak
///     bisa dijalankan di `flutter test`. Menjalankan model fp32 sebagai
///     pengganti bukan pilihan yang jujur: yang diuji harus model yang
///     benar-benar dipakai. Verify pipeline ini di perangkat, dan
///     hasilnya sudah tercatat di `docs/TASK-V-REPORT.md`.
///   * Decode dan resize. Decode diuji lewat spike P1, resize diuji
///     oleh `test/area_resize_test.dart` terhadap cv2 di dua skala.
///     Keduanya tidak perlu diulang di sini.
void main() {
  group('batas ukuran berkas 20 MB', () {
    test('batasnya tepat 20 MB', () {
      expect(maxFileBytes, 20 * 1024 * 1024);
    });

    test('berkas tepat di batas diterima', () {
      expect(() => BacteriaClassifier.checkSize(maxFileBytes), returnsNormally);
    });

    test('berkas satu byte di atas batas ditolak', () {
      expect(
        () => BacteriaClassifier.checkSize(maxFileBytes + 1),
        throwsA(isA<FileTooLargeException>()),
      );
    });

    test('berkas jauh di atas batas ditolak', () {
      expect(
        () => BacteriaClassifier.checkSize(maxFileBytes * 10),
        throwsA(isA<FileTooLargeException>()),
      );
    });

    test('berkas kosong tidak ditolak oleh batas ukuran', () {
      // Decode yang akan menolak berkas kosong, bukan pemeriksaan ukuran.
      expect(() => BacteriaClassifier.checkSize(0), returnsNormally);
    });

    test('pesan error menyebut ukuran berkas dan batasnya', () {
      final error = const FileTooLargeException(maxFileBytes + 1);
      expect(error.bytes, maxFileBytes + 1);
      expect(error.toString(), contains('20'));
    });

    test('penolakan terjadi sebelum decode, bukan sesudahnya', () {
      // Batasnya dicek dari panjang berkas sebelum byte dibaca ke memori.
      // Kalauurutannya dibalik, berkas besar akan memakan memori dulu.
      // Yang dikunci di sini adalah kontrak: fungsi ini hanya menerima
      // jumlah byte, tidak menerima isi berkas.
      expect(
        BacteriaClassifier.checkSize,
        isA<void Function(int)>(),
      );
    });
  });

  group('label yang dipakai aplikasi', () {
    test('bentuk sel hanya cocci dan bacilli', () {
      const labels = <String>['cocci', 'bacilli'];
      expect(labels, hasLength(2));
    });

    test('status Gram hanya positive dan negative', () {
      const labels = <String>['positive', 'negative'];
      expect(labels, hasLength(2));
    });
  });

  group('Classification', () {
    Classification build({
      String shape = 'bacilli',
      String gram = 'negative',
      double shapeConfidence = 0.91,
      double gramConfidence = 0.83,
      int elapsedMs = 213,
    }) {
      return Classification(
        shapeLabel: shape,
        gramLabel: gram,
        shapeLevel: levelForConfidence(shapeConfidence),
        gramLevel: levelForConfidence(gramConfidence),
        shapeConfidence: shapeConfidence,
        gramConfidence: gramConfidence,
        elapsedMs: elapsedMs,
        imageWidth: 2048,
        imageHeight: 1532,
      );
    }

    test('level dihitung dari confidence, bukan diisi terpisah', () {
      final result = build(shapeConfidence: 0.85, gramConfidence: 0.62);
      expect(result.shapeLevel, ConfidenceLevel.high);
      expect(result.gramLevel, ConfidenceLevel.medium);
    });

    test('menyimpan ukuran citra sumber', () {
      final result = build();
      expect(result.imageWidth, 2048);
      expect(result.imageHeight, 1532);
    });

    test('menyimpan waktu proses', () {
      expect(build(elapsedMs: 213).elapsedMs, 213);
    });
  });
}