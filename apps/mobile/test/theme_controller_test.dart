import 'package:bacteriacv_mobile/controllers/theme_controller.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('tema awal mengikuti sistem', () {
    expect(ThemeController().value, ThemeMode.system);
  });

  test('select mengganti mode dan memberi tahu pendengar', () {
    final ThemeController c = ThemeController();
    int calls = 0;
    c.addListener(() => calls++);
    c.select(ThemeMode.dark);
    expect(c.value, ThemeMode.dark);
    expect(calls, 1);
  });
}
