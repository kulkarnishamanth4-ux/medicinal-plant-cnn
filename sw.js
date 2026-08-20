const CACHE_NAME = 'herbscan-v1';
const ASSETS_TO_CACHE = [
  './offline_scanner.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
  './class_names.json',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest',
  './tfjs_model/tfjs_model/model.json',
  './tfjs_model/tfjs_model/group1-shard1of3.bin',
  './tfjs_model/tfjs_model/group1-shard2of3.bin',
  './tfjs_model/tfjs_model/group1-shard3of3.bin'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(ASSETS_TO_CACHE);
      })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});
