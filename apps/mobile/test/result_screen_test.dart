import 'package:bacteriacv_mobile/models/classifier.dart';
import 'package:bacteriacv_mobile/views/disclaimer.dart';
import 'package:bacteriacv_mobile/views/result_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

Classification _result({
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

Future<void> _pump(WidgetTester tester, Classification result) async {
  await tester.pumpWidget(
    MaterialApp(home: ResultScreen(result: result)),
  );
  await tester.pumpAndSettle();
}

void main() {
  group('disclaimer', () {
    testWidgets('memuat kalimat wajib dengan persis', (
      WidgetTester tester,
    ) async {
      await _pump(tester, _result());
      for (final String sentence in Disclaimer.sentences) {
        expect(find.text(sentence), findsOneWidget, reason: sentence);
      }
    });

    test('kalimat utama Google Play tidak boleh diubah', () {
      // Persis seperti yang disepakati untuk Google Play. Menambah
      // atau mengubah kalimat ini melanggar kesepakatan, jadi test ini
      // mengunci teksnya, bukan hanya keberadaannya.
      expect(
        primaryDisclaimer,
        'This app is not a medical device and does not diagnose, treat, '
        'or prevent any condition.',
      );
    });

    test('ada empat kalimat wajib', () {
      expect(Disclaimer.sentences, hasLength(4));
    });

    testWidgets('terlihat tanpa menggulir', (WidgetTester tester) async {
      await _pump(tester, _result());
      // Disclaimer ada di anak pertama ListView, sebelum hasil apa pun,
      // jadi tidak perlu menggulir untuk membacanya.
      expect(find.byType(Disclaimer), findsOneWidget);
      final box = tester.getRect(find.byType(Disclaimer));
      expect(box.top, lessThan(200));
      expect(Disclaimer.sentences.first, primaryDisclaimer);
    });
  });

  group('hasil klasifikasi', () {
    testWidgets('menampilkan bentuk sel dan terjemahannya', (
      WidgetTester tester,
    ) async {
      await _pump(tester, _result(shape: 'bacilli'));
      expect(find.text('bacilli (batang)'), findsOneWidget);
    });

    testWidgets('menampilkan bentuk cocci dengan terjemahan', (
      WidgetTester tester,
    ) async {
      await _pump(tester, _result(shape: 'cocci'));
      expect(find.text('cocci (kokoid)'), findsOneWidget);
    });

    testWidgets('menampilkan status Gram dan terjemahannya', (
      WidgetTester tester,
    ) async {
      await _pump(tester, _result(gram: 'negative'));
      expect(find.text('negative (negatif)'), findsOneWidget);
    });

    testWidgets('menampilkan level sebagai teks, bukan hanya warna', (
      WidgetTester tester,
    ) async {
      await _pump(
        tester,
        _result(shapeConfidence: 0.85, gramConfidence: 0.62),
      );
      expect(find.text('high'), findsOneWidget);
      expect(find.text('medium'), findsOneWidget);
      expect(find.text('low'), findsNothing);
    });

    testWidgets('level low tampil sebagai teks', (WidgetTester tester) async {
      await _pump(
        tester,
        _result(shapeConfidence: 0.42, gramConfidence: 0.11),
      );
      expect(find.text('low'), findsNWidgets(2));
    });

    testWidgets('menampilkan waktu proses dalam milidetik', (
      WidgetTester tester,
    ) async {
      await _pump(tester, _result(elapsedMs: 213));
      expect(find.text('213 ms'), findsOneWidget);
    });

    testWidgets('menampilkan ukuran citra sumber', (WidgetTester tester) async {
      await _pump(tester, _result());
      expect(find.text('2048 x 1532'), findsOneWidget);
    });
  });

  group('confidence tidak ditampilkan sebagai persen', () {
    testWidgets('tidak ada tanda persen di layar hasil', (
      WidgetTester tester,
    ) async {
      await _pump(tester, _result());
      expect(find.textContaining('%'), findsNothing);
    });

    testWidgets('angka confidence tidak muncul sebagai teks', (
      WidgetTester tester,
    ) async {
      // 0,91 sebagai confidence akan muncul sebagai "91" kalau
      // ditampilkan sebagai persen. Angka itu tidak boleh ada.
      await _pump(tester, _result(shapeConfidence: 0.91, gramConfidence: 0.83));
      expect(find.textContaining('91'), findsNothing);
      expect(find.textContaining('83'), findsNothing);
    });

    testWidgets('confidence 1,0 tidak bocor sebagai angka', (
      WidgetTester tester,
    ) async {
      await _pump(
        tester,
        _result(shapeConfidence: 1.0, gramConfidence: 1.0),
      );
      expect(find.textContaining('100'), findsNothing);
    });
  });

  group('tidak ada klaim diagnosis', () {
    testWidgets('layar hasil tidak memuat kata diagnose', (
      WidgetTester tester,
    ) async {
      await _pump(tester, _result());
      // Kata "diagnose" muncul di disclaimer sebagai penyangkal, dan itu
      // wajib. Yang dicek di sini: tidak ada klaim bahwa hasil ini
      // mendiagnosis.
      final finder = find.byWidgetPredicate(
        (Widget w) =>
            w is Text &&
            w.data != null &&
            w.data!.toLowerCase().contains('diagnos') &&
            !w.data!.contains(primaryDisclaimer) &&
            !w.data!.contains(aidDisclaimer),
      );
      expect(finder, findsNothing);
    });
  });
}