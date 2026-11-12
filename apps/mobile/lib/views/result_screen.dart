import 'package:flutter/material.dart';

import '../models/classifier.dart';
import 'disclaimer.dart';

/// Layar hasil klasifikasi.
///
/// Yang ditampilkan: bentuk sel, status Gram beserta levelnya, dan waktu
/// proses dalam milidetik.
///
/// Yang SENGAJA tidak ditampilkan: confidence sebagai persen. Model int8
/// meleset sampai 0,542656 terhadap acuan float32 meskipun F1 tetap 1,000000,
/// jadi angkanya tidak terkalibrasi. Level high, medium, atau low adalah
/// yang ditampilkan, sama seperti aplikasi web.
class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key, required this.result});

  /// Hasil yang ditampilkan.
  final Classification result;

  /// Terjemahan label bentuk ke Bahasa Indonesia, ditampilkan berdampingan
  /// dengan label aslinya supaya tidak ada ambiguitas.
  static const Map<String, String> shapeTranslations = <String, String>{
    'cocci': 'kokoid',
    'bacilli': 'batang',
  };

  static const Map<String, String> gramTranslations = <String, String>{
    'positive': 'positif',
    'negative': 'negatif',
  };

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final shapeId =
        shapeTranslations[result.shapeLabel] ?? result.shapeLabel;
    final gramId = gramTranslations[result.gramLabel] ?? result.gramLabel;

    return Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: <Widget>[
          // Disclaimer di paling atas, terlihat tanpa menggulir.
          const Disclaimer(),
          const SizedBox(height: 16),
          Text('Cell shape', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          _Row(
            label: 'Result',
            value: '${result.shapeLabel} ($shapeId)',
          ),
          _Row(
            label: 'Confidence level',
            value: result.shapeLevel.wireName,
          ),
          const SizedBox(height: 20),
          Text('Gram status', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          _Row(
            label: 'Result',
            value: '${result.gramLabel} ($gramId)',
          ),
          _Row(
            label: 'Confidence level',
            value: result.gramLevel.wireName,
          ),
          const SizedBox(height: 20),
          Text('Processing', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          _Row(label: 'Elapsed', value: '${result.elapsedMs} ms'),
          _Row(
            label: 'Source image',
            value: '${result.imageWidth} x ${result.imageHeight}',
          ),
          const SizedBox(height: 20),
          Text(
            'Confidence is not shown as a percentage. The int8 model keeps '
            'F1 at 1.0000 on 66 DIBaS test images, but its confidence values '
            'are not calibrated, so a percentage would mislead.',
            style: theme.textTheme.bodySmall,
          ),
        ],
      ),
    );
  }
}

/// Satu baris label dan nilai. Nilai adalah teks, bukan hanya warna, jadi
/// status tidak bergantung pada warna saja.
class _Row extends StatelessWidget {
  const _Row({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Semantics(
      label: '$label: $value',
      excludeSemantics: true,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            SizedBox(
              width: 150,
              child: Text(label, style: theme.textTheme.bodyMedium),
            ),
            Expanded(
              child: Text(
                value,
                style: theme.textTheme.bodyLarge?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}