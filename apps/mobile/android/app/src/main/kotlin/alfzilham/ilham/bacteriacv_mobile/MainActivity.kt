package alfzilham.ilham.bacteriacv_mobile

import io.flutter.embedding.android.FlutterActivity

/**
 * Host aplikasi.
 *
 * Tidak ada Intent dari adb shell, tidak ada mode pengukuran, dan tidak
 * ada MethodChannel. Semua itu milik spike dan sudah dihapus: laporan
 * pengukuran diambil dari berkas lewat `adb shell am start -a runHeadless`,
 * dan jalur itu tidak boleh ada di aplikasi yang dipakai orang, karena
 * ia bisa memicu proses dari luar tanpa lewat antarmuka.
 *
 * Inferensi berjalan sepenuhnya di perangkat. Tidak ada server, tidak
 * ada API, tidak ada telemetry.
 */
class MainActivity : FlutterActivity()