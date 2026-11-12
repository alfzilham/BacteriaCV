import 'package:flutter/material.dart';

/// Kalimat yang WAJIB persis seperti tertulis di sini.
///
/// Kalimat pertama adalah rumusan yang disepakati untuk Google Play dan
/// tidak boleh ditambah, diubah, atau dipotong. Kalimat lain menjaga
/// hasil tetap dibaca sebagai bantuan, bukan diagnosis.
const String primaryDisclaimer =
    'This app is not a medical device and does not diagnose, treat, '
    'or prevent any condition.';

const String segmentationDisclaimer =
    'Segmentation groups cells, it does not isolate individual cells.';

const String spiralDisclaimer =
    'The spiral class is unpopulated and is excluded from the metrics.';

const String aidDisclaimer = 'This result is an aid, not a diagnosis.';

/// Disclaimer wajib pada layar hasil.
///
/// Ditampilkan di bagian atas layar, sebelum hasil apa pun, supaya
/// terlihat tanpa harus menggulir.
class Disclaimer extends StatelessWidget {
  const Disclaimer({super.key});

  /// Semua kalimat disclaimer, untuk diuji.
  static const List<String> sentences = <String>[
    primaryDisclaimer,
    segmentationDisclaimer,
    spiralDisclaimer,
    aidDisclaimer,
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: theme.colorScheme.outline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              Icon(
                Icons.info_outline,
                size: 18,
                color: theme.colorScheme.onSurfaceVariant,
              ),
              const SizedBox(width: 6),
              Text('Disclaimer', style: theme.textTheme.labelLarge),
            ],
          ),
          const SizedBox(height: 6),
          // Semantik disclaimer: dibaca pembaca layar sebagai catatan
          // penting, bukan teks dekoratif.
          Semantics(
            container: true,
            label: 'Disclaimer. ${sentences.join(' ')}',
            child: ExcludeSemantics(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  for (final String sentence in sentences)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Text(
                        sentence,
                        style: theme.textTheme.bodySmall,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}