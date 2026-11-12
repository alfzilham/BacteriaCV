import 'dart:convert';
import 'dart:io';

import 'package:bacteriacv_mobile/models/area_resize.dart';
import 'package:flutter_test/flutter_test.dart';

/// Menguji `resizeArea` di lib/models/area_resize.dart terhadap hasil
/// `cv2.INTER_AREA`.
///
/// Kenapa ini harus ada: INTER_AREA tidak punya operator padanan di ONNX
/// maupun TFLite yang builtin, jadi harus diimplementasikan ulang di Dart.
/// Kalau implementasi Dart itu berbeda dari cv2, seluruh perbandingan P2
/// dan P3 mengukur hal yang salah - dan salahnya tidak akan kelihatan,
/// karena model tetap mengembalikan dua angka yang masuk akal.
///
/// Acuan dihasilkan oleh spike/scripts/compare_pytorch_fp16.py, bukan
/// ditulis manual. Kalau berkasnya hilang, tes ini SKIP dan menyebutkan
/// begitu, bukan lolos diam-diam.
void main() {
  // Nama stem berkas acuan yang dihasilkan compare_pytorch_fp16.py.
  const stems = <String>['contoh-1-bacilli-positif', 'sintetis-2048x1532'];

  for (final stem in stems) {
    test('resizeArea cocok dengan cv2 INTER_AREA untuk $stem', () {
      final artifacts = Directory('spike/artifacts');
      final sourceFile = File('${artifacts.path}/resize_src_$stem.bin');
      final referenceFile = File('${artifacts.path}/resize_ref_$stem.bin');
      final metaFile = File('${artifacts.path}/resize_ref_$stem.json');

      // Kasus di bawah memakai citra 768 x 574, bukan ukuran DIBaS asli
      // 2048 x 1532. Skala turunnya berbeda, dan jendela INTER_AREA
      // bergantung pada skala, jadi ini bukti parsial. Test kedua
      // menutup kekurangan itu dengan sumber sintetis 2048 x 1532 yang
      // skalnya sama dengan DIBaS asli.

      if (!sourceFile.existsSync() ||
          !referenceFile.existsSync() ||
          !metaFile.existsSync()) {
        markTestSkipped(
          'acuan belum ada. Jalankan: python spike/scripts/'
          'compare_pytorch_fp16.py',
        );
        return;
      }

      final meta = (jsonDecode(metaFile.readAsStringSync()) as Map)
          .cast<String, Object?>();
      final sourceWidth = meta['source_width']! as int;
      final sourceHeight = meta['source_height']! as int;
      final target = meta['target']! as int;

      final source = sourceFile.readAsBytesSync();
      final reference = referenceFile.readAsBytesSync();

      expect(
        source.length,
        sourceWidth * sourceHeight * 3,
        reason: 'sumber harus RGB uint8 dengan urutan HWC',
      );
      expect(reference.length, target * target * 3);

      final actual = resizeArea(source, sourceWidth, sourceHeight, target);
      expect(actual.length, reference.length);

      // Selisih per piksel dihitung, bukan hanya "tidak crash". cv2
      // membulatkan hasil rata-rata dengan pembulatan setengah ke atas,
      // sedangkan implementasi Dart memakai round() yang untuk nilai
      // negatif bisa meleset satu satuan. Jadi toleransi satu, dan test
      // ini juga melaporkan berapa banyak byte yang berbeda supaya
      // penyimpangan systematic terlihat, bukan cuma yang terburuk.
      var worstDelta = 0;
      var worstIndex = -1;
      var deltaCount = 0;
      for (var i = 0; i < reference.length; i++) {
        final delta = (actual[i] - reference[i]).abs();
        if (delta != 0) {
          deltaCount++;
          if (delta > worstDelta) {
            worstDelta = delta;
            worstIndex = i;
          }
        }
      }

      expect(
        worstDelta,
        lessThanOrEqualTo(1),
        reason: 'selisih terburuk di byte $worstIndex: '
            'dart ${actual[worstIndex]} vs cv2 ${reference[worstIndex]}; '
            'total $deltaCount byte berbeda dari ${reference.length}',
      );

      // Bookmark supaya tidak ada yang menaikkan toleransi tanpa sadar.
      expect(
        deltaCount / reference.length,
        lessThan(0.01),
        reason: 'lebih dari 1 persen byte berbeda. Itu bukan pembulatan, '
            'itu perbedaan algoritma.',
      );
    });
  }
}