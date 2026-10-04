# Review meowCDDAI — GPT × Gemini — AI, screening và dữ liệu lâm sàng

**Ngày:** 2026-09-28  
**Phạm vi:** Chỉ `meowCDDAI/` và các thư mục con.  
**Mục tiêu:** Review tiếp các boundary của FastAPI AI service: secret/config, ASQ scoring thiếu dữ liệu, auth/ownership child, LLM output, vector/database consistency và async jobs.

> **Trạng thái:** `MỞ — A-01 ĐÃ XÁC NHẬN QUA SOURCE; A-02 ĐẾN A-05 CẦN CHẠY TRỰC TIẾP VÀ GEMINI PHẢN BIỆN`.
>
> Service này trả kết quả screening, AI recommendation, clinical scribe, gaze/speech/pose signals và nội dung can thiệp. Kết quả AI không được trình bày như chẩn đoán nếu thiếu dữ liệu, provenance hoặc human review.

---

## 1. Bối cảnh hệ thống

`meowCDDAI` là FastAPI/Python service, entry point [main.py](main.py), mặc định port 8102. Nó mount:

- Qdrant/Postgres vector/search và upload routes;
- extraction và async job routes;
- ASQ-3 hybrid AI;
- gaze, expression, pose, interaction, speech screening;
- worksheet, social story, telemetry, clinical scribe và child harness.

Các test application hiện có chủ yếu mock boundary DB và HTTP tới meowAI. Điều đó hữu ích cho logic orchestration nhưng chưa chứng minh deployment auth, credential, schema database/vector hoặc model artifact production.

---

## 2. Finding đã xác nhận qua source inspection

### A-01 / P0 — Credential mặc định của 9router bị commit trong config

**Vị trí:** [config.py:26-29](config.py#L26-L29).

`NINE_ROUTER_API_KEY` có một giá trị mặc định dạng secret trong source thay vì bắt buộc lấy từ secret manager/environment. Báo cáo không lặp lại giá trị credential cụ thể.

### Tác động

- Secret có thể bị lộ qua git, image, log debug hoặc artifact CI.
- Người có key có thể gọi LLM gateway dưới quota/tài khoản của hệ thống.
- Local fallback dễ bị deploy nhầm lên staging/production.

### Cách sửa bắt buộc

- Revoke/rotate credential hiện tại ngay; kiểm tra git history, Docker layers và CI logs.
- Production fail-fast nếu `NINE_ROUTER_API_KEY` thiếu; không dùng default key.
- Tách local/test secret khỏi source và dùng secret manager/CI secret.
- Redact Authorization/header trong log, exception và tracing.
- Thêm startup/config test phân biệt dev với production.

**Verdict:** `CONFIRMED — P0 — CHẶN DEPLOY/ROTATE NGAY`.

---

## 3. Findings cần GPT/Gemini phản biện

### A-02 / P1 — Missing ASQ input có thể bị biến thành điểm thuận lợi và kết luận sai

**Vị trí:** [asq_ai_service.py:225-249](asq_ai_service.py#L225-L249), [asq_ai_service.py:279-292](asq_ai_service.py#L279-L292).

Service hiện dùng default khi payload thiếu:

- `age_interval_months` mặc định 12;
- domain score trong normative loop mặc định `60.0`;
- một số anomaly/ML feature dùng default 50 hoặc 60;
- nếu không có black/gray zone, flow có thể trả `TYPICAL` và risk thấp.

Đây là **review lead mạnh cần chạy output trực tiếp**, chưa ghi là confirmed runtime cho tới khi gọi service với payload rỗng/partial/invalid.

### Tác động cần xác minh

Payload mất score vì client lỗi, schema drift hoặc request rỗng có thể bị diễn giải thành trẻ đạt chuẩn thay vì `INSUFFICIENT_DATA`. Đây là false reassurance trong luồng screening.

### Cách sửa bắt buộc nếu tái hiện được

- Pydantic/request validation bắt buộc age và đủ domain scores theo contract.
- Missing/null/invalid score phải trả trạng thái `INSUFFICIENT_DATA`, không chạy recommendation TYPICAL.
- Tách `not provided` khỏi score cao; không dùng 60 như sentinel có nghĩa clinical.
- Ghi `data_quality`, `missing_fields`, model/norm source và confidence thực tế.
- Không kích hoạt intervention hoặc “đạt chuẩn” từ input chưa đủ.

**Case bắt buộc:** `{}`, chỉ có age, một domain, null/negative/non-numeric score, tuổi ngoài norms, premature thiếu tuổi; đối chiếu `overall_status`, risk, recommendation, domain result và HTTP status.

**Trạng thái:** `CHƯA XÁC MINH — cần output trực tiếp`.

### A-03 / P1 — Endpoint child harness/clinical không thể hiện auth/ownership dependency

**Vị trí:** [child_harness.py:34-132](routers/child_harness.py#L34-L132), [clinical_scribe.py:39-65](routers/clinical_scribe.py#L39-L65).

Các route nhận `child_id` và trả profile/context/clinical summary nhưng file router hiện không thể hiện dependency xác thực hoặc ownership check. Có thể deployment đặt gateway ở ngoài, vì vậy đây là finding cần kiểm tra end-to-end, không kết luận chỉ từ file router.

### Tác động cần xác minh

- User có thể đọc hoặc cập nhật context của child khác nếu service được expose trực tiếp.
- Prompt snippet, diagnosis, trigger, mastered/target skills và episode data là dữ liệu nhạy cảm.
- Clinical report có thể tạo output lâm sàng từ `child_id` do caller tự chọn.

### Case bắt buộc

- không Authorization;
- token user A với child B;
- role parent/teacher/admin khác nhau;
- GET context/profile và POST update/add-memory/sync-test-result;
- clinical summary trỏ child không thuộc caller;
- prompt injection trong `dominant_interests`, `triggers_to_avoid`, `recent_episodes`.

Expected: 401/403 hoặc policy rõ; không đọc/ghi chéo child, không đưa raw prompt injection vào system prompt không được kiểm soát.

**Trạng thái:** `CHƯA XÁC MINH — cần kiểm tra gateway và service trực tiếp`.

### A-04 / P1 — LLM/clinical report boundary chưa chứng minh schema, provenance và safe wording

**Vị trí:** [llm_client.py:21-65](llm_client.py#L21-L65), [clinical_scribe.py:39-65](routers/clinical_scribe.py#L39-L65), service được import tại `services/clinical_scribe_service.py`.

`call_meow_ai` gửi payload OpenAI-compatible, nhận `choices[0].message.content` và trả chuỗi. Review cần kiểm tra phần service parser và response contract:

- empty choices/content có bị biến thành report hợp lệ không;
- timeout/502 có retry hoặc ghi job partial không;
- response không đúng schema có bị trả 200 không;
- clinical recommendation có đánh dấu draft/AI-generated, source và uncertainty không;
- user input/LLM HTML có được sanitize trước khi frontend render không;
- log có chứa child name, episode, score hoặc Authorization không.

**Case bắt buộc:** meowAI 200 empty choices, malformed JSON, 401/429/5xx, timeout 600s, prompt chứa HTML/script, low-confidence response và response thiếu source. Expected là lỗi typed/low-confidence rõ, không tạo diagnosis xác quyết hoặc HTML nguy hiểm.

**Trạng thái:** `CHƯA XÁC MINH`.

### A-05 / P1 — E2E async pipeline mock boundary chưa chứng minh production integration

**Vị trí:** [test_e2e_pipeline.py:1-22](tests/test_e2e_pipeline.py#L1-L22), [job_runner.py](job_runner.py), [job_repository.py](job_repository.py).

Test E2E hiện mô tả và kiểm chứng job lifecycle bằng in-memory repository, mock HTTP tới meowAI. Nó tốt cho property pending → processing → completed/failed, nhưng không chứng minh:

- Postgres transaction/connection failure và concurrent workers;
- meowAI thật, auth, timeout và schema drift;
- Qdrant/Postgres consistency sau upload/extraction;
- model artifact/scaler version trong ASQ;
- resume/poll sau process restart;
- duplicate job và idempotency.

**Cách bổ sung:** contract test staging cho từng boundary, DB failure injection, process restart/resume, duplicate request, meowAI 401/429/5xx, Qdrant write failure sau Postgres commit và ngược lại. Ghi rõ mock/live trong report.

**Trạng thái:** `CHƯA XÁC MINH — limitation của test coverage, chưa tự động là bug`.

### A-06 / P1 cần xác minh — CORS/public router surface và telemetry PII

**Vị trí:** [main.py:20-33](main.py#L20-L33), [main.py:36-75](main.py#L36-L75), [routers/telemetry.py](routers/telemetry.py).

CORS bật `allow_credentials=True` với allowlist từ env hoặc local defaults, và service mount nhiều router. Cần endpoint matrix để xác minh route nào có auth, rate limit và PII redaction; đặc biệt telemetry, clinical, child harness, upload/admin và screening media.

**Case bắt buộc:** request không token, origin lạ, origin local trong production, upload file quá lớn/sai MIME, telemetry chứa child ID/voice metadata, admin route từ user thường.

**Trạng thái:** `CHƯA XÁC MINH`.

---

## 4. Bảng yêu cầu Gemini phản biện

| Finding | Case bắt buộc | Verdict Gemini | Output thực tế |
|---|---|---|---|
| A-01 secret | secret scan, startup prod thiếu key, rotate/history/image | Chưa phản biện |  |
| A-02 ASQ missing data | empty/partial/invalid/out-of-range payload | Chưa phản biện |  |
| A-03 child auth | unauthenticated và cross-child GET/POST | Chưa phản biện |  |
| A-04 LLM boundary | empty/malformed/timeout/5xx/prompt injection | Chưa phản biện |  |
| A-05 async integration | live DB/meowAI, restart, duplicate, boundary failure | Chưa phản biện |  |
| A-06 CORS/PII | origin, upload, telemetry, admin/public route matrix | Chưa phản biện |  |

Gemini phải ưu tiên response chạy từ code hiện tại, kiểm tra trực tiếp service/gateway, không suy ra auth chỉ từ frontend hoặc Docker network.

---

## 5. Test và verify hiện tại

### Test hiện có

- `tests/` có job runner/repository/routes, speech assessment, child harness, worksheet, social story, clinical scribe, meltdown warning và E2E orchestration.
- `ai_enhanced_gaze_tracking/tests/` có nhiều test về quality, confidence fallback, fusion, calibration, error handling và compatibility.
- `docs/TESTING_GUIDE.md` mô tả các nhóm lệnh nhưng không thay cho output của lần chạy hiện tại.

### Lệnh bắt buộc

```bash
pytest -q
pytest -q tests/test_asq_ai_service.py tests/test_child_harness_api.py tests/test_clinical_scribe_api.py
pytest -q tests/test_e2e_pipeline.py tests/test_job_runner.py tests/test_routes_jobs.py
pytest -q ai_enhanced_gaze_tracking/tests
```

Đối với test cần model/GPU/DB, phải ghi rõ dependency được mock, skipped hay live. Không ghi “full suite pass” nếu một nhóm bị bỏ qua do dependency.

### Kết quả hiện tại

- **Chưa chạy test mới trong review này.**
- Không có cơ sở ghi coverage hoặc full-suite pass.
- A-01 là security incident độc lập với trạng thái test.

---

## 6. Tiêu chí đóng review

Chỉ đóng khi:

1. A-01 đã revoke/rotate, không còn credential default trong source/history/image; production fail-fast.
2. A-02 trả `INSUFFICIENT_DATA` cho empty/partial/invalid payload và không tạo recommendation thuận lợi giả.
3. A-03 có auth + ownership server-side cho mọi child-sensitive route.
4. A-04 validate LLM response, sanitize output, lưu provenance/confidence và dùng wording draft/non-diagnostic.
5. A-05 có contract/integration test cho DB, meowAI, Qdrant, restart và duplicate.
6. A-06 có route/CORS/rate-limit/PII matrix trong profile production.
7. `pytest` và các nhóm test liên quan chạy lại với output được lưu, nêu rõ skipped/failure.

**Kết luận hiện tại:** `CHƯA ĐƯỢC ĐÓNG REVIEW — A-01 P0 ĐÃ XÁC NHẬN; A-02 ĐẾN A-06 CHỜ GEMINI CHẠY, PHẢN BIỆN VÀ SỬA.`
