import 'package:flutter/material.dart';

import 'controllers/theme_controller.dart';
import 'theme/tokens.dart';
import 'views/upload_screen.dart';

/// Akar aplikasi.
///
/// Satu bahasa dulu, English. Tidak ada login, tidak ada akun, tidak
/// ada telemetry, tidak ada server. Semua inferensi berjalan di
/// perangkat dan data pengguna tidak dikirim ke mana pun.
class BacteriaCvApp extends StatelessWidget {
  const BacteriaCvApp({super.key});

  /// Membangun tema untuk terang dan gelap dari token desain.
  ///
  /// Dipisah supaya `test/accessibility_test.dart` bisa mengukur kontras
  /// pada tema yang BENAR-BENAR dipakai aplikasi, bukan salinannya di
  /// dalam test. Kalau tema berubah dan test masih hijau, test itu tidak
  /// mengukur apa pun.
  static ThemeData themeFor(Brightness brightness) {
    final bool dark = brightness == Brightness.dark;
    final Color paper = dark ? Tokens.darkPaper : Tokens.paper;
    final Color ink = dark ? Tokens.darkInk : Tokens.ink;
    final Color tile = dark ? Tokens.darkTile : Tokens.tile;
    final ColorScheme scheme = ColorScheme(
      brightness: brightness,
      primary: ink,
      onPrimary: paper,
      secondary: dark ? Tokens.darkViolet : Tokens.violet,
      onSecondary: paper,
      error: dark ? Tokens.darkError : Tokens.error,
      onError: paper,
      surface: paper,
      onSurface: ink,
      onSurfaceVariant: dark ? Tokens.darkText2 : Tokens.text2,
      surfaceContainerHighest: tile,
      outline: dark ? Tokens.darkEdge : Tokens.edge,
      outlineVariant: dark ? Tokens.darkLine : Tokens.line,
    );
    return ThemeData(
      colorScheme: scheme,
      scaffoldBackgroundColor: paper,
      useMaterial3: true,
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size.fromHeight(56),
          shape: const StadiumBorder(),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          minimumSize: const Size.fromHeight(56),
          shape: const StadiumBorder(),
          side: BorderSide(color: ink),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: ThemeController.instance,
      builder: (BuildContext context, ThemeMode mode, Widget? _) {
        return MaterialApp(
          title: 'BacteriaCV',
          debugShowCheckedModeBanner: false,
          theme: themeFor(Brightness.light),
          darkTheme: themeFor(Brightness.dark),
          themeMode: mode,
          home: const UploadScreen(),
        );
      },
    );
  }
}
