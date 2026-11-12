import 'dart:math' as math;

import 'package:bacteriacv_mobile/app.dart';
import 'package:bacteriacv_mobile/models/classifier.dart';
import 'package:bacteriacv_mobile/views/result_screen.dart';
import 'package:bacteriacv_mobile/views/upload_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

/// Mengukur kontras dan target sentuh pada tema yang benar-benar dipakai.
///
/// Ambang dari `docs/DESIGN.md` versi web:
///   * teks di atas latar minimal 4,5:1 (WCAG 2.1 AA)
///   * elemen bukan teks minimal 3:1 (WCAG 1.4.11)
/// Dan target sentuh minimal 48x48 logical pixel, angka yang diminta brief
/// Task Z1. Versi web menyebut 44px; 48 yang diuji karena itu standar
/// Android dan itu yang diminta.
///
/// Mengukur, bukan menebak. Warna diambil dari
/// `BacteriaCvApp.themeFor`, fungsi yang sama dengan yang dipakai
/// `MaterialApp`, jadi kalau tema berubah angka di sini ikut berubah.
/// Diuji pada mode terang dan gelap karena perangkat bisa salah satu.
double contrastRatio(Color a, Color b) {
  final double la = _luminance(a);
  final double lb = _luminance(b);
  final double hi = math.max(la, lb);
  final double lo = math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}

double _luminance(Color c) {
  double channel(double v) {
    return v <= 0.03928
        ? v / 12.92
        : math.pow((v + 0.055) / 1.055, 2.4).toDouble();
  }

  return 0.2126 * channel(c.r) +
      0.7152 * channel(c.g) +
      0.0722 * channel(c.b);
}

Classification _result({
  ConfidenceLevel shape = ConfidenceLevel.high,
  ConfidenceLevel gram = ConfidenceLevel.medium,
}) {
  return Classification(
    shapeLabel: 'bacilli',
    gramLabel: 'negative',
    shapeLevel: shape,
    gramLevel: gram,
    shapeConfidence: 0.91,
    gramConfidence: 0.65,
    elapsedMs: 287,
    imageWidth: 2048,
    imageHeight: 1532,
  );
}

void _report(String label, double ratio, double threshold) {
  // ignore: avoid_print
  print(
    '$label = ${ratio.toStringAsFixed(2)}:1 '
    '(ambang ${threshold.toStringAsFixed(1)}:1) '
    '${ratio >= threshold ? "LOLOS" : "GAGAL"}',
  );
}

void main() {
  for (final Brightness brightness in Brightness.values) {
    final String name = brightness == Brightness.light ? 'light' : 'dark';
    final ColorScheme c = BacteriaCvApp.themeFor(brightness).colorScheme;

    group('kontras $name', () {
      test('teks utama di atas surface minimal 4,5:1', () {
        final double ratio = contrastRatio(c.onSurface, c.surface);
        _report('$name onSurface/surface', ratio, 4.5);
        expect(ratio, greaterThanOrEqualTo(4.5));
      });

      test('label tombol di atas primary minimal 4,5:1', () {
        final double ratio = contrastRatio(c.onPrimary, c.primary);
        _report('$name onPrimary/primary', ratio, 4.5);
        expect(ratio, greaterThanOrEqualTo(4.5));
      });

      test('teks disclaimer di atas latar kotak minimal 4,5:1', () {
        final double ratio = contrastRatio(
          c.onSurfaceVariant,
          c.surfaceContainerHighest,
        );
        _report('$name disclaimer text', ratio, 4.5);
        expect(ratio, greaterThanOrEqualTo(4.5));
      });

      test('teks error di atas surface minimal 4,5:1', () {
        final double ratio = contrastRatio(c.error, c.surface);
        _report('$name error/surface', ratio, 4.5);
        expect(ratio, greaterThanOrEqualTo(4.5));
      });

      test('batas kotak disclaimer minimal 3:1 sebagai elemen bukan teks', () {
        final double ratio = contrastRatio(
          c.outline,
          c.surfaceContainerHighest,
        );
        _report('$name border', ratio, 3.0);
        expect(ratio, greaterThanOrEqualTo(3.0));
      });

      test('latar kotak disclaimer terhadap halaman: angka dicatat', () {
        // Latar kotak ini hanya pemelompokan, bukan komponen yang
        // harus dikenali untuk memakai aplikasi, dan bukan elemen yang
        // Border itulah yang menandai batas kotak.
        // adalah border, dan border-nya diuji 3:1 di test sebelumnya.
        //
        // Angka latar tetap dicetak supaya tidak ada yang mengira
        // kotak ini kontrasnya memadai.
        final double ratio = contrastRatio(
          c.surfaceContainerHighest,
          c.surface,
        );
        // ignore: avoid_print
        print(
          '$name box background = ${ratio.toStringAsFixed(2)}:1 '
          '(dekoratif, TIDAK diklaim memenuhi 3:1)',
        );
        expect(ratio, greaterThan(0.0));
      });
    });
  }

  group('target sentuh minimal 48x48', () {
    const double minimum = 48;

    testWidgets('tombol Choose file', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: BacteriaCvApp.themeFor(Brightness.light),
          home: const UploadScreen(),
        ),
      );
      await tester.pumpAndSettle();
      final Size size = tester.getSize(find.byType(FilledButton).first);
      // ignore: avoid_print
      print('Choose file = ${size.width} x ${size.height}');
      expect(size.height, greaterThanOrEqualTo(minimum));
      expect(size.width, greaterThanOrEqualTo(minimum));
    });

    testWidgets('tombol Analyze', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: BacteriaCvApp.themeFor(Brightness.light),
          home: const UploadScreen(),
        ),
      );
      await tester.pumpAndSettle();
      final Size size = tester.getSize(find.byType(FilledButton).last);
      // ignore: avoid_print
      print('Analyze = ${size.width} x ${size.height}');
      expect(size.height, greaterThanOrEqualTo(minimum));
      expect(size.width, greaterThanOrEqualTo(minimum));
    });

    testWidgets('target sentuh tombol Pilih file mencakup seluruh tombol', (
      WidgetTester tester,
    ) async {
      // Yang diukur adalah target yang ditekan, yaitu tombolnya, bukan
        // glyph ikon di dalamnya. Ikon 18x18 tidak punya target sendiri
        // dan tidak boleh diuji seolah-olah punya.
        await tester.pumpWidget(
          MaterialApp(
            theme: BacteriaCvApp.themeFor(Brightness.light),
            home: const UploadScreen(),
          ),
        );
        await tester.pumpAndSettle();
        final Size button = tester.getSize(find.byType(FilledButton).first);
        final Size icon = tester.getSize(find.byIcon(Icons.folder_open));
        // ignore: avoid_print
        print(
          'target tombol ${button.width} x ${button.height}, '
          'ikon ${icon.width} x ${icon.height} di dalamnya',
        );
        expect(button.height, greaterThanOrEqualTo(minimum));
        expect(icon.height, lessThan(button.height));
      });
  });

  group('level status sebagai teks, bukan hanya warna', () {
    testWidgets('high, medium, low tampil sebagai teks', (
      WidgetTester tester,
    ) async {
      for (final ConfidenceLevel level in ConfidenceLevel.values) {
        await tester.pumpWidget(
          MaterialApp(
            theme: BacteriaCvApp.themeFor(Brightness.light),
            home: ResultScreen(
              result: _result(shape: level, gram: level),
            ),
          ),
        );
        await tester.pumpAndSettle();
        expect(find.text(level.wireName), findsNWidgets(2));
      }
    });

    testWidgets('level punya label semantik untuk pembaca layar', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: BacteriaCvApp.themeFor(Brightness.light),
          home: ResultScreen(result: _result()),
        ),
      );
      await tester.pumpAndSettle();
      // Nilai level ikut terbaca, jadi pengguna pembaca layar tidak
      // hanya mendengar kata "Confidence level" tanpa jawabannya.
      expect(find.bySemanticsLabel('Confidence level: high'), findsOneWidget);
      expect(
        find.bySemanticsLabel('Confidence level: medium'),
        findsOneWidget,
      );
    });

    testWidgets('hasil dan waktu proses punya label semantik', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: BacteriaCvApp.themeFor(Brightness.light),
          home: ResultScreen(result: _result()),
        ),
      );
      await tester.pumpAndSettle();
      expect(
        find.bySemanticsLabel('Result: bacilli (batang)'),
        findsOneWidget,
      );
      expect(find.bySemanticsLabel('Elapsed: 287 ms'), findsOneWidget);
    });

    testWidgets('disclaimer punya satu label semantik utuh', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: BacteriaCvApp.themeFor(Brightness.light),
          home: ResultScreen(result: _result()),
        ),
      );
      await tester.pumpAndSettle();
      expect(
        find.bySemanticsLabel(RegExp('not a medical device')),
        findsOneWidget,
      );
    });
  });
}