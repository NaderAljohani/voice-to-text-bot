SUPPORTED_FORMATS_TEXT = "OGG, MP3, WAV, M4A, MP4, WEBM, AAC, FLAC"


def format_seconds(seconds: int) -> str:
    minutes = seconds // 60
    remainder = seconds % 60
    if minutes and remainder:
        return f"{minutes} دقيقة و {remainder} ثانية"
    if minutes:
        return f"{minutes} دقيقة"
    return f"{remainder} ثانية"


def format_minutes_from_seconds(seconds: int) -> str:
    minutes = seconds // 60
    remainder = seconds % 60
    if remainder:
        return f"{minutes}.{round(remainder / 60 * 10)}"
    return str(minutes)


START = (
    "أهلاً بك في بوت التفريغ الصوتي.\n\n"
    "أرسل رسالة صوتية أو ملفاً صوتياً أو فيديو يحتوي على صوت، وسأحوّله إلى نص مكتوب.\n"
    "يدعم البوت العربية والإنجليزية، ويمكنه التعامل مع المقاطع المختلطة قدر الإمكان.\n\n"
    "رصيد التجربة المجانية: 10 دقائق يومياً.\n\n"
    "تنبيه: قد يتم حفظ سجل الاستخدام والنصوص الناتجة لأغراض الأمان ومنع إساءة الاستخدام وتحسين الخدمة."
)

RECEIVING_FILE = "تم استلام الملف. جارٍ فحص مدة الصوت..."
PROCESSING_STARTED = "بدأت عملية التفريغ. قد يستغرق ذلك بضع دقائق حسب طول الصوت."
TRANSCRIPTION_COMPLETED = "تم التفريغ بنجاح ✅"
RAW_TRANSCRIPT_CAPTION = "النص الكامل."
TIMESTAMPED_TRANSCRIPT_CAPTION = "النص الكامل مع توقيت المقاطع."
ERROR_OCCURRED = "حدث خطأ أثناء المعالجة. تأكد من أن الملف صوتي صالح ثم حاول مرة أخرى."
NO_CLEAR_SPEECH = "لم أتمكن من استخراج كلام واضح من الملف. الرجاء إرسال ملف بصوت أوضح."
UNSUPPORTED_MESSAGE = "أرسل رسالة صوتية أو ملفاً صوتياً أو فيديو يحتوي على صوت."
EMPTY_TRANSCRIPT = "لم يتم العثور على كلام واضح في الملف."
BANNED_USER = "تم إيقاف استخدامك للبوت بسبب مخالفة شروط الاستخدام."
ADMIN_ONLY = "هذا الأمر مخصص للمشرفين فقط."
ADMIN_USAGE_BAN = "الاستخدام: /ban <telegram_id> <reason>"
ADMIN_USAGE_UNBAN = "الاستخدام: /unban <telegram_id>"
ADMIN_USAGE_USERLOGS = "الاستخدام: /userlogs <telegram_id>"
ADMIN_USAGE_USER = "الاستخدام: /user <telegram_id>"
ADMIN_USAGE_ADDMINUTES = "الاستخدام: /addminutes <telegram_id> <minutes>"
ADMIN_USAGE_SETBALANCE = "الاستخدام: /setbalance <telegram_id> <minutes>"
ADMIN_USAGE_LOGS = "الاستخدام: /logs"
ADMIN_USAGE_ADMIN = "الاستخدام: /admin"
USAGE_BALANCE = "الاستخدام: /balance"
USAGE_MYID = "الاستخدام: /myid"
ADMIN_NO_LOGS = "لا توجد سجلات حالياً."
GPU_SETUP_ERROR_PUBLIC = "حدث خطأ تقني أثناء معالجة الملف. جرّب مرة أخرى لاحقًا، أو أرسل ملفًا أقصر."
GPU_SETUP_ERROR_ADMIN = "إعدادات كرت الشاشة غير مكتملة. راجع سجلات التشغيل."
ADMIN_BYPASS_BALANCE_MESSAGE = "وضع الأدمن مفعّل — لم يتم خصم الرصيد."
FILE_TOO_LONG_LIMIT = "مدة الملف أطول من الحد المسموح. الرجاء إرسال ملف أقصر."
VIDEO_HAS_NO_AUDIO = "لم يتم العثور على مسار صوتي داخل الفيديو."
USER_ALREADY_PROCESSING = "لديك ملف قيد المعالجة حالياً. انتظر حتى يكتمل."
FREE_QUOTA_ENDED = "انتهى رصيدك المجاني لهذا اليوم. يمكنك المحاولة غداً."
INSUFFICIENT_BALANCE_FOR_FILE = (
    "مدة الملف تتجاوز رصيدك المتاح حالياً. الرجاء إرسال ملف أقصر أو المحاولة لاحقاً."
)
UNSUPPORTED_FILE_FORMAT = (
    "صيغة الملف غير مدعومة.\n\n"
    "الصيغ المدعومة:\n"
    f"{SUPPORTED_FORMATS_TEXT}\n\n"
    "يمكنك إرسال رسالة صوتية، ملف صوتي، أو فيديو يحتوي على صوت."
)

ADMIN_PANEL = (
    "لوحة تحكم الأدمن:\n\n"
    "/logs - عرض آخر العمليات\n"
    "/userlogs <telegram_id> - عرض عمليات مستخدم\n"
    "/user <telegram_id> - عرض معلومات مستخدم\n"
    "/addminutes <telegram_id> <minutes> - إضافة رصيد إضافي لمستخدم\n"
    "/setbalance <telegram_id> <minutes> - تعيين الرصيد الإضافي لمستخدم\n"
    "/ban <telegram_id> <reason> - حظر مستخدم\n"
    "/unban <telegram_id> - فك حظر مستخدم\n"
    "/myid - عرض Telegram ID الخاص بك\n"
    "/balance - عرض رصيدك\n\n"
    "ملاحظة: إذا كان ADMIN_BYPASS_BALANCE=true فلن يتم خصم الرصيد من حساب الأدمن."
)


def daily_balance_message(
    daily_remaining_seconds: int,
    daily_free_minutes: int,
    extra_balance_seconds: int,
) -> str:
    daily_remaining = format_minutes_from_seconds(daily_remaining_seconds)
    if extra_balance_seconds > 0:
        extra_minutes = format_minutes_from_seconds(extra_balance_seconds)
        return (
            f"رصيدك اليومي المتبقي: {daily_remaining} من {daily_free_minutes} دقائق.\n"
            f"الرصيد الإضافي: {extra_minutes} دقيقة."
        )
    return f"رصيدك اليومي المتبقي: {daily_remaining} من {daily_free_minutes} دقائق."
