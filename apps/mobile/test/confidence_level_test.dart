import 'package:bacteriacv_mobile/models/classifier.dart';
import 'package:flutter_test/flutter_test.dart';

/// Mengunci ambang level terhadap nilai yang sama dipakai aplikasi web.
///
/// Tujuannya bukan menguji classifier, tapi mengunci KONSTANTA. Kalau
/// ambangnya berubah, test ini gagal, dan itu memang yang diinginkan:
/// aplikasi ini dan aplikasi web harus memberi keputusan level yang sama
/// untuk confidence yang sama. Dua aplikasi berbeda untuk model yang sama
/// adalah bug, bukan penyesuaian.
void main() {
  group('ambang level terkunci', () {
    test('high adalah 0.8 dan medium adalah 0.6', () {
      expect(highThreshold, 0.8);
      expect(mediumThreshold, 0.6);
    });

    test('kategori level hanya high, medium, low', () {
      expect(
        ConfidenceLevel.values.map((ConfidenceLevel v) => v.wireName).toList(),
        <String>['high', 'medium', 'low'],
      );
    });
  });

  group('levelForConfidence', () {
    test('high untuk confidence di atas dan tepat di 0.8', () {
      expect(levelForConfidence(1.0), ConfidenceLevel.high);
      expect(levelForConfidence(0.95), ConfidenceLevel.high);
      expect(levelForConfidence(0.8), ConfidenceLevel.high);
    });

    test('medium tepat di 0.6 dan di bawah 0.8', () {
      expect(levelForConfidence(0.6), ConfidenceLevel.medium);
      expect(levelForConfidence(0.7), ConfidenceLevel.medium);
      expect(levelForConfidence(0.79999), ConfidenceLevel.medium);
    });

    test('low di bawah 0.6', () {
      expect(levelForConfidence(0.0), ConfidenceLevel.low);
      expect(levelForConfidence(0.5), ConfidenceLevel.low);
      expect(levelForConfidence(0.59999), ConfidenceLevel.low);
    });

    test('batas 0.8 tidak ikut turun ke medium', () {
      // Pernah jadi bug: memakai operator > sehingga 0.8 jatuh ke medium.
      // Runner web memakai >=, jadi 0.8 harus high.
      expect(levelForConfidence(0.8), isNot(ConfidenceLevel.medium));
    });

    test('batas 0.6 tidak ikut turun ke low', () {
      expect(levelForConfidence(0.6), isNot(ConfidenceLevel.low));
    });

    test('1.0 dan 0.0 adalah kedua ujung', () {
      expect(levelForConfidence(1.0), ConfidenceLevel.high);
      expect(levelForConfidence(0.0), ConfidenceLevel.low);
    });
  });
}