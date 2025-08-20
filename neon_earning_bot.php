<?php
declare(strict_types=1);

// Neon Earning Bot - Minimal CLI stub
// Safe defaults: reads only environment variables; no filesystem writes

function log_line(string $level, string $message): void {
    $ts = date('c');
    fwrite(STDERR, "$ts | $level | neon_earning_bot | $message\n");
}

function getEnvOrDefault(string $key, string $default): string {
    $val = getenv($key);
    return $val !== false ? $val : $default;
}

$interval = (int) getEnvOrDefault('PHP_BOT_INTERVAL_SECONDS', '60');
$endpoint = getEnvOrDefault('PHP_BOT_ENDPOINT', '');

log_line('INFO', 'PHP bot started. interval='.$interval.'s');

while (true) {
    try {
        if ($endpoint !== '') {
            // Optional: call an endpoint (e.g., your FastAPI service)
            $ch = curl_init($endpoint);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_TIMEOUT, 5);
            $response = curl_exec($ch);
            if ($response === false) {
                log_line('WARN', 'Request failed: '.curl_error($ch));
            } else {
                log_line('INFO', 'Endpoint response length='.strlen($response));
            }
            curl_close($ch);
        } else {
            log_line('INFO', 'Heartbeat tick');
        }
    } catch (Throwable $e) {
        log_line('ERROR', $e->getMessage());
    }
    sleep($interval);
}