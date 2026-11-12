import 'package:flutter/material.dart';

/// Pilihan tema aplikasi. Awalnya mengikuti sistem (ThemeMode.system).
///
/// Pilihan hanya disimpan selama aplikasi berjalan. Menyimpannya antar
/// peluncuran butuh paket penyimpanan tambahan, dan itu belum diputuskan.
class ThemeController extends ValueNotifier<ThemeMode> {
  ThemeController() : super(ThemeMode.system);

  /// Instans tunggal yang dipakai MaterialApp dan tombol tema.
  static final ThemeController instance = ThemeController();

  /// Mengganti mode tema.
  void select(ThemeMode mode) => value = mode;
}
