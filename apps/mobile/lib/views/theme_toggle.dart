import 'package:flutter/material.dart';

import '../controllers/theme_controller.dart';

/// Pemilih tema tiga pilihan: terang, gelap, sistem.
class ThemeToggle extends StatelessWidget {
  const ThemeToggle({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: ThemeController.instance,
      builder: (BuildContext context, ThemeMode mode, Widget? _) {
        return SegmentedButton<ThemeMode>(
          showSelectedIcon: false,
          segments: const <ButtonSegment<ThemeMode>>[
            ButtonSegment<ThemeMode>(
              value: ThemeMode.light,
              icon: Icon(Icons.light_mode_outlined),
              tooltip: 'Light',
            ),
            ButtonSegment<ThemeMode>(
              value: ThemeMode.dark,
              icon: Icon(Icons.dark_mode_outlined),
              tooltip: 'Dark',
            ),
            ButtonSegment<ThemeMode>(
              value: ThemeMode.system,
              icon: Icon(Icons.brightness_auto_outlined),
              tooltip: 'System',
            ),
          ],
          selected: <ThemeMode>{mode},
          onSelectionChanged: (Set<ThemeMode> s) =>
              ThemeController.instance.select(s.first),
        );
      },
    );
  }
}
