import pandas as pd

data = [
    # FAQ
    {"category": "FAQ", "tag": "Account", "question": "Bagaimana cara mereset password akun saya?", "answer": "Anda dapat mereset password dengan mengklik tombol 'Lupa Password' pada halaman login. Tautan reset password akan dikirim ke email terdaftar Anda."},
    {"category": "FAQ", "tag": "Account", "question": "Bagaimana cara mengubah alamat email saya?", "answer": "Anda bisa mengubah alamat email melalui menu 'Pengaturan Akun', lalu pilih 'Ubah Email' dan ikuti instruksi yang diberikan."},
    {"category": "FAQ", "tag": "Payment", "question": "Metode pembayaran apa saja yang diterima?", "answer": "Kami menerima kartu kredit/debit (Visa, MasterCard), transfer bank (BCA, Mandiri, BNI, BRI), dan e-wallet (GoPay, OVO, Dana)."},
    {"category": "FAQ", "tag": "Shipping", "question": "Berapa lama waktu pengiriman pesanan?", "answer": "Waktu pengiriman standar memakan waktu 2-3 hari kerja untuk wilayah Jabodetabek, dan 3-7 hari kerja untuk luar pulau."},
    {"category": "FAQ", "tag": "Shipping", "question": "Apakah saya bisa melacak pesanan saya?", "answer": "Ya, Anda dapat melacak pesanan dengan memasukkan nomor resi pada menu 'Lacak Pesanan' di website atau aplikasi kami."},
    {"category": "FAQ", "tag": "Account", "question": "Bagaimana cara menghapus akun saya?", "answer": "Untuk menghapus akun, silakan hubungi tim dukungan pelanggan kami melalui email, dan proses akan memakan waktu hingga 30 hari."},
    {"category": "FAQ", "tag": "Payment", "question": "Mengapa pembayaran saya ditolak?", "answer": "Pembayaran bisa ditolak karena saldo tidak mencukupi, informasi kartu salah, atau batas limit transaksi telah tercapai. Silakan hubungi bank Anda."},
    {"category": "FAQ", "tag": "Product", "question": "Apakah produk memiliki garansi?", "answer": "Ya, sebagian besar produk elektronik kami memiliki garansi resmi 1 tahun dari produsen."},
    {"category": "FAQ", "tag": "Account", "question": "Bolehkah saya memiliki lebih dari satu akun?", "answer": "Sesuai kebijakan kami, satu pengguna hanya diperbolehkan memiliki satu akun utama. Akun duplikat dapat dibekukan."},
    {"category": "FAQ", "tag": "Promo", "question": "Bagaimana cara menggunakan kode promo?", "answer": "Kode promo dapat dimasukkan pada halaman checkout sebelum Anda menyelesaikan pembayaran."},
    {"category": "FAQ", "tag": "Promo", "question": "Mengapa kode promo saya tidak bisa digunakan?", "answer": "Kode promo mungkin sudah kadaluarsa, kuota telah habis, atau tidak memenuhi syarat minimum pembelian."},
    {"category": "FAQ", "tag": "Order", "question": "Bisakah saya membatalkan pesanan?", "answer": "Pesanan hanya dapat dibatalkan jika statusnya belum 'Diproses'. Anda bisa melakukannya melalui menu 'Pesanan Saya'."},
    {"category": "FAQ", "tag": "Order", "question": "Bagaimana cara mengubah alamat pengiriman pesanan?", "answer": "Alamat pengiriman tidak bisa diubah jika pesanan sudah diproses. Jika belum, Anda dapat membatalkan pesanan dan membuat pesanan baru."},
    {"category": "FAQ", "tag": "Product", "question": "Apakah saya bisa memesan barang yang sedang kosong?", "answer": "Anda bisa menggunakan fitur 'Beritahu Saya' agar mendapat notifikasi saat barang kembali tersedia."},

    # Policy
    {"category": "Policy", "tag": "Return", "question": "Apa kebijakan pengembalian barang?", "answer": "Barang dapat dikembalikan dalam waktu 7 hari sejak diterima, asalkan masih dalam kondisi asli dan segel belum terbuka."},
    {"category": "Policy", "tag": "Refund", "question": "Bagaimana proses pengembalian dana (refund)?", "answer": "Pengembalian dana akan diproses ke metode pembayaran awal dalam waktu 3-5 hari kerja setelah barang retur kami terima dan periksa."},
    {"category": "Policy", "tag": "Privacy", "question": "Apakah data pribadi saya aman?", "answer": "Kami menggunakan enkripsi standar industri (SSL) untuk melindungi data pribadi Anda dan tidak akan membagikannya kepada pihak ketiga tanpa izin."},
    {"category": "Policy", "tag": "Terms", "question": "Apa syarat dan ketentuan penggunaan layanan?", "answer": "Syarat dan ketentuan lengkap dapat dibaca pada halaman 'Terms and Conditions' di bagian bawah website kami."},
    {"category": "Policy", "tag": "Shipping", "question": "Apakah ada kebijakan pengiriman gratis?", "answer": "Pengiriman gratis berlaku untuk pesanan di atas Rp 500.000 untuk wilayah Jabodetabek."},
    {"category": "Policy", "tag": "Warranty", "question": "Apa yang tidak dicakup dalam garansi?", "answer": "Garansi tidak mencakup kerusakan akibat kelalaian pengguna, bencana alam, atau modifikasi tidak resmi."},
    {"category": "Policy", "tag": "Return", "question": "Siapa yang menanggung biaya ongkir pengembalian?", "answer": "Biaya ongkir pengembalian ditanggung oleh pembeli, kecuali jika barang yang diterima rusak atau tidak sesuai pesanan."},
    {"category": "Policy", "tag": "Privacy", "question": "Bagaimana cara meminta penghapusan data saya?", "answer": "Permintaan penghapusan data (Right to be Forgotten) dapat diajukan dengan mengirim email ke privacy@perusahaan.com."},
    {"category": "Policy", "tag": "Order", "question": "Apakah perusahaan berhak membatalkan pesanan sepihak?", "answer": "Kami berhak membatalkan pesanan jika terindikasi adanya penipuan, kesalahan harga sistem, atau stok habis secara tidak terduga."},
    {"category": "Policy", "tag": "Review", "question": "Apa kebijakan moderasi ulasan produk?", "answer": "Ulasan yang mengandung kata kasar, spam, atau tidak relevan dengan produk akan dihapus oleh moderator kami."},
    {"category": "Policy", "tag": "Affiliate", "question": "Bagaimana kebijakan program afiliasi?", "answer": "Komisi afiliasi diberikan sebesar 5% dari setiap penjualan sukses melalui link unik Anda, dibayarkan setiap tanggal 15."},

    # Troubleshooting
    {"category": "Troubleshooting", "tag": "Login", "question": "Saya tidak bisa login, muncul pesan 'Kredensial Tidak Valid'.", "answer": "Pastikan email dan password yang Anda masukkan benar. Perhatikan penggunaan huruf kapital. Jika masih gagal, gunakan fitur 'Lupa Password'."},
    {"category": "Troubleshooting", "tag": "App", "question": "Aplikasi sering crash atau keluar sendiri.", "answer": "Cobalah untuk meng-update aplikasi ke versi terbaru, hapus cache aplikasi, atau restart perangkat Anda."},
    {"category": "Troubleshooting", "tag": "Payment", "question": "Saldo sudah terpotong tapi status pesanan masih 'Menunggu Pembayaran'.", "answer": "Mohon tunggu hingga 15 menit. Jika status tidak berubah, kirimkan bukti transfer ke Customer Service kami untuk pengecekan manual."},
    {"category": "Troubleshooting", "tag": "Web", "question": "Halaman website tidak bisa dimuat (blank white screen).", "answer": "Silakan bersihkan cache dan cookies browser Anda, atau coba buka website menggunakan mode incognito (penyamaran)."},
    {"category": "Troubleshooting", "tag": "Promo", "question": "Voucher gratis ongkir tidak memotong biaya pengiriman.", "answer": "Pastikan Anda sudah memilih ekspedisi yang mendukung voucher tersebut dan memenuhi syarat minimum belanja."},
    {"category": "Troubleshooting", "tag": "Account", "question": "Saya tidak menerima email verifikasi.", "answer": "Cek folder Spam atau Junk di email Anda. Jika tidak ada, klik tombol 'Kirim Ulang Email Verifikasi' di halaman profil."},
    {"category": "Troubleshooting", "tag": "Product", "question": "Produk elektronik yang baru dibeli tidak bisa menyala.", "answer": "Pastikan kabel power terhubung dengan benar dan stopkontak berfungsi. Jika masih tidak menyala, hubungi CS untuk proses retur barang cacat (DOA)."},
    {"category": "Troubleshooting", "tag": "App", "question": "Fitur GPS tidak mendeteksi lokasi saya dengan benar.", "answer": "Pastikan Anda telah memberikan izin lokasi pada aplikasi. Pergi ke Pengaturan HP > Aplikasi > Izin > aktifkan Lokasi."},
    {"category": "Troubleshooting", "tag": "Order", "question": "Nomor resi pengiriman tidak bisa dilacak.", "answer": "Data resi biasanya butuh waktu 1x24 jam untuk terupdate di sistem kurir. Silakan coba lacak kembali besok hari."},
    {"category": "Troubleshooting", "tag": "Web", "question": "Tombol 'Beli Sekarang' tidak bisa diklik.", "answer": "Kemungkinan stok barang sedang dikunci karena ada pembeli lain yang sedang di halaman checkout, atau Anda perlu menyegarkan halaman."},
    {"category": "Troubleshooting", "tag": "Account", "question": "Akun saya tiba-tiba terkunci.", "answer": "Akun terkunci biasanya karena terlalu banyak percobaan login yang salah. Tunggu 30 menit atau hubungi CS untuk verifikasi identitas."},
    {"category": "Troubleshooting", "tag": "App", "question": "Notifikasi aplikasi tidak masuk di HP saya.", "answer": "Buka pengaturan notifikasi di HP Anda, pastikan izin notifikasi untuk aplikasi kami dalam keadaan aktif."},
    {"category": "Troubleshooting", "tag": "Payment", "question": "Gagal melakukan pembayaran dengan kartu kredit (Error 3D Secure).", "answer": "Pastikan nomor HP yang terdaftar di bank aktif untuk menerima SMS OTP. Jika tidak masuk, hubungi bank penerbit kartu."},
    {"category": "Troubleshooting", "tag": "Review", "question": "Saya tidak bisa mengunggah foto pada ulasan produk.", "answer": "Pastikan ukuran foto tidak melebihi 5MB dan format foto adalah JPG atau PNG."},

    # Contact Information
    {"category": "Contact Information", "tag": "CS", "question": "Bagaimana cara menghubungi Customer Service?", "answer": "Anda bisa menghubungi CS kami melalui Live Chat di aplikasi (24/7), email ke support@perusahaan.com, atau telepon ke 1500-123."},
    {"category": "Contact Information", "tag": "CS", "question": "Berapa jam operasional Call Center?", "answer": "Call Center kami (1500-123) beroperasi setiap hari Senin hingga Minggu, pukul 08:00 - 20:00 WIB."},
    {"category": "Contact Information", "tag": "Social", "question": "Apa saja akun media sosial resmi perusahaan?", "answer": "Instagram: @perusahaan_id, Twitter: @perusahaan_care, Facebook: Perusahaan Indonesia."},
    {"category": "Contact Information", "tag": "Office", "question": "Di mana alamat kantor pusat perusahaan?", "answer": "Kantor pusat kami berlokasi di Gedung Sudirman Tower, Lt. 15, Jl. Jend. Sudirman Kav 21, Jakarta Selatan 12920."},
    {"category": "Contact Information", "tag": "Business", "question": "Ke mana saya harus mengirim proposal kerjasama bisnis?", "answer": "Untuk keperluan B2B dan partnership, silakan kirimkan email dan proposal Anda ke partnership@perusahaan.com."},
    {"category": "Contact Information", "tag": "Career", "question": "Di mana saya bisa melihat info lowongan kerja?", "answer": "Informasi lowongan kerja terbaru dapat dilihat di halaman karir kami di karir.perusahaan.com atau akun LinkedIn resmi perusahaan."},
    {"category": "Contact Information", "tag": "Press", "question": "Kontak untuk kebutuhan media dan press release?", "answer": "Rekan-rekan media dapat menghubungi tim Public Relations kami melalui email di pr@perusahaan.com."},
    {"category": "Contact Information", "tag": "Store", "question": "Apakah perusahaan memiliki toko offline?", "answer": "Saat ini kami beroperasi 100% secara online dan belum memiliki toko fisik (offline store)."},
    {"category": "Contact Information", "tag": "CS", "question": "Berapa lama SLA balasan email CS?", "answer": "Tim CS kami berkomitmen untuk membalas semua email masuk dalam waktu maksimal 1x24 jam kerja."},
    {"category": "Contact Information", "tag": "Complaint", "question": "Bagaimana prosedur eskalasi keluhan tingkat lanjut?", "answer": "Jika solusi dari CS dirasa kurang memuaskan, Anda bisa membalas tiket email dengan subject [ESKALASI], maka tim Supervisor akan menanganinya."},
    {"category": "Contact Information", "tag": "Feedback", "question": "Di mana saya bisa memberikan saran atau masukan untuk aplikasi?", "answer": "Kami sangat menghargai masukan Anda. Silakan isi form feedback di menu 'Bantuan' > 'Beri Masukan'."}
]

df = pd.DataFrame(data)
df.to_csv('data/knowledge_base.csv', index=False)
print("Created knowledge_base.csv with 50 entries.")
