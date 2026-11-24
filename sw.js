// オフラインでも開けるように、アプリのファイルをブラウザに保存する(音声は保存しない)
const C = 'utamix-10d1591775c0', F = ["./", "index.html", "dsp.js", "mp3enc.js", "flacenc.js", "rec-worklet.js", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png", "icons/icon-180.png"];
self.addEventListener('install', e => { e.waitUntil(caches.open(C).then(c => c.addAll(F)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== C).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => { if (e.request.method !== 'GET') return; e.respondWith(caches.match(e.request, { ignoreSearch: true }).then(r => r || fetch(e.request))); });
