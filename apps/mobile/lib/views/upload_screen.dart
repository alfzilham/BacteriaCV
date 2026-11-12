import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../models/classifier.dart';
import 'result_screen.dart';
import 'theme_toggle.dart';

/// Layar pertama: pilih berkas, lihat pratinjau, lalu analisis.
///
/// Memilih BERKAS, bukan foto dari galeri. Data DIBaS adalah TIFF dan
/// file TIFF tidak muncul di galeri Android, jadi aplikasi yang hanya
/// bisa mengambil foto galeri tidak bisa dipakai untuk data yang
/// dilatih modelnya.
class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  String? _path;
  String? _name;
  int? _bytes;
  Uint8List? _previewJpeg;
  bool _busy = false;
  String? _error;

  Future<void> _pick() async {
    final picked = await FilePicker.pickFiles(
      // TIFF tidak ada di galeri Android, jadi jenis berkas
      // dideklarasikan eksplisit.
      type: FileType.custom,
      allowedExtensions: const <String>[
        'tif',
        'tiff',
        'png',
        'jpg',
        'jpeg',
      ],
    );
    if (picked.isEmpty) {
      return;
    }
    final pickedFile = picked.first;
    final path = pickedFile.path;
    if (path == null) {
      setState(() => _error = 'The selected file is not on local storage.');
      return;
    }

    // Batas ukuran diperiksa sebelum byte dibaca ke memori, supaya
    // berkas besar ditolak tanpa sempat memakan RAM. Diperiksa juga
    // lagi di dalam classifier, sebelum decode.
    final int? reported = pickedFile.lengthSync();
    final size = reported ?? 0;
    try {
      BacteriaClassifier.checkSize(size);
    } on FileTooLargeException catch (error) {
      setState(() => _error = '$error');
      return;
    }

    // Pratinjau hanya untuk tampilan. Decode untuk klasifikasi
    // dilakukan di dalam classifier, bukan hasil yang dipakai di sini,
    // supaya tidak ada dua jalur decode yang bisa berbeda.
    Uint8List? preview;
    try {
      final bytes = await File(path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded != null) {
        preview = Uint8List.fromList(
          img.encodeJpg(img.copyResize(decoded, width: 640), quality: 80),
        );
      }
    } on Object {
      // Pratinjau adalah tambahan. Kalau gagal, analisis tetap boleh
      // dicoba dan pipeline yang menentukan hasilnya.
      preview = null;
    }

    if (!mounted) {
      return;
    }
    setState(() {
      _path = path;
      _name = pickedFile.name;
      _bytes = size;
      _previewJpeg = preview;
      _error = null;
    });
  }

  Future<void> _analyze() async {
    final path = _path;
    if (path == null) {
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final classifier = await BacteriaClassifier.instance();
      final result = await classifier.classify(path);
      if (!mounted) {
        return;
      }
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (BuildContext context) => ResultScreen(result: result),
        ),
      );
    } on FileTooLargeException catch (error) {
      if (mounted) {
        setState(() => _error = '$error');
      }
    } on Object catch (error) {
      if (mounted) {
        setState(() => _error = '$error');
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ready = _path != null && !_busy;

    return Scaffold(
      appBar: AppBar(
        title: const Text('BacteriaCV'),
        actions: const <Widget>[ThemeToggle(), SizedBox(width: 12)],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: <Widget>[
          Text(
            'Choose a microscopy image to classify. TIFF, PNG and JPEG are '
            'accepted, up to 20 MB.',
            style: theme.textTheme.bodyMedium,
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _busy ? null : _pick,
            icon: const Icon(Icons.folder_open),
            label: const Text('Choose file'),
          ),
          const SizedBox(height: 16),
          if (_previewJpeg != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.memory(
                _previewJpeg!,
                key: const Key('preview'),
                fit: BoxFit.contain,
              ),
            )
          else
            Container(
              height: 180,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                'No image selected',
                style: theme.textTheme.bodySmall,
              ),
            ),
          const SizedBox(height: 12),
          if (_name != null)
            Text(
              '$_name, ${((_bytes ?? 0) / 1024).toStringAsFixed(0)} KB',
              style: theme.textTheme.bodySmall,
            ),
          if (_error != null) ...<Widget>[
            const SizedBox(height: 12),
            Semantics(
              liveRegion: true,
              child: Text(
                _error!,
                key: const Key('error'),
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ),
          ],
          const SizedBox(height: 24),
          FilledButton(
            // Aktif hanya setelah gambar dipilih.
            onPressed: ready ? _analyze : null,
            child: Text(_busy ? 'Analyzing...' : 'Analyze'),
          ),
          const SizedBox(height: 24),
          Text(
            'Internal validation: macro F1 1.0000 for both heads on 66 DIBaS '
            'test images. This is internal validation, not external '
            'validation.',
            style: theme.textTheme.bodySmall,
          ),
        ],
      ),
    );
  }
}