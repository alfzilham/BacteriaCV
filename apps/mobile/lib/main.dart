import 'package:flutter/material.dart';

import 'app.dart';

/// Entry point aplikasi.
///
/// Tidak ada Intent dari adb shell, tidak ada mode pengukuran, tidak ada
/// MethodChannel. Semua alat ukur spike sudah dihapus; yang tersisa
/// hanya pipeline yang sudah divalidasi dan antarmukanya.
void main() {
  runApp(const BacteriaCvApp());
}