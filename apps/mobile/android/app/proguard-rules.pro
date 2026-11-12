# Aturan ProGuard/R8 untuk build release.
#
# ONNX Runtime Android memanggil kelas lewat JNI dari native. R8 tidak
# bisa melihat pemanggilan itu dari sisi Java, jadi tanpa aturan keep
# class ONNX Runtime akan di-strip atau di-rename dan session akan gagal
# saat runtime dengan error yang tidak jelas.
#
# Berkas ini WAJIB ada dan WAJIB ditulis sebagai ASCII tanpa BOM.
# R8 gagal membaca berkas ini dengan "Input length = 1" kalau ada BOM,
# dan gejalanya muncul sebagai kegagalan minifyReleaseWithR8 yang
# tidak menyangka hubungannya dengan encoding.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**