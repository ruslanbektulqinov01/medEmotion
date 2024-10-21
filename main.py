import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters.command import Command
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, select, func, and_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Constants and Configuration
BOT_TOKEN = os.getenv('BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
MAX_DAILY_CONSULTATIONS = 10
CONSULTATION_TIMEOUT = 300  # 5 minutes in seconds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database initialization
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# Database models
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    username = Column(String)
    phone_number = Column(String)
    language_code = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime)
    consultation_count = Column(Integer, default=0)
    daily_consultation_count = Column(Integer, default=0)
    last_consultation_date = Column(DateTime)
    is_blocked = Column(Boolean, default=False)
    feedback_score = Column(Float, default=0.0)
    feedback_count = Column(Integer, default=0)


class Consultation(Base):
    __tablename__ = 'consultations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    category = Column(String)
    content = Column(String, nullable=False)
    response = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    is_resolved = Column(Boolean, default=False)
    feedback_score = Column(Integer)
    feedback_text = Column(String)


# States for conversation management
class ConsultationStates(StatesGroup):
    category_selection = State()
    conversation = State()


# Initialize bot with FSM storage
storage = MemoryStorage()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=storage)


# AI response generator
class AIResponseGenerator:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.context = ("Siz malakali shifokorsiz va tibbiy konsultatsiya boti sifatida ishlaysiz. "
                        "Foydalanuvchilarga umumiy tibbiy maslahatlar bering, shu bilan birga aniq tashxis qo'yish "
                        "va davolanish uchun ularni haqiqiy tibbiyot mutaxassislariga murojaat qilishga undab turing. "
                        "Doim professional va g'amxo'rlik ohangida javob bering. Oldingi suhbat tarixini inobatga olgan "
                        "holda javoblarni shakllantiring. O'zbek tilida javob bering.")

    async def generate_response(self, category: str, context: str) -> str:
        try:
            category_prompts = {
                'general': "Foydalanuvchining umumiy tibbiy holati haqidagi savoliga malakali shifokor sifatida javob bering: ",
                'medicine': "Dori-darmonlar haqida malakali shifokor sifatida ma'lumot bering, ularning maqsadi, yon ta'siri va dozasi haqida ma'lumot bering, ammo o'z-o'zini davolashni tavsiya etmang: ",
                'hospitals': "Kasalxonalar va tibbiy muassasalar haqida malakali shifokor sifatida ma'lumot bering, ularning ixtisosligi, manzili va aloqa ma'lumotlarini taqdim eting: ",
                'specialists': "Tibbiyot mutaxassislari haqida malakali shifokor sifatida ma'lumot bering, ularning ixtisosligi, tajribasi va aloqa ma'lumotlarini taqdim eting: "}

            prompt = f"{self.context}\n{category_prompts.get(category, '')}\n{context}"
            response = await asyncio.to_thread(lambda: self.model.generate_content(prompt).text)
            disclaimer = ("\n\n⚠️ Eslatma: Ushbu ma'lumot faqat umumiy maslahat uchun. "
                          "Aniq tashxis va davolanish uchun shifokorga murojaat qiling.")
            return f"{response}{disclaimer}"
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return "Kechirasiz, texnik nosozlik yuz berdi. Iltimos, keyinroq urinib ko'ring."


ai_generator = AIResponseGenerator()


# Database manager
class DatabaseManager:
    @staticmethod
    async def init_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @staticmethod
    async def get_user(user_id: int) -> Optional[User]:
        async with async_session() as session:
            result = await session.execute(select(User).where(User.user_id == user_id))
            return result.scalar_one_or_none()

    @staticmethod
    async def add_or_update_user(user: types.User) -> User:
        async with async_session() as session:
            db_user = await DatabaseManager.get_user(user.id)
            if db_user is None:
                db_user = User(user_id=user.id, first_name=user.first_name, last_name=user.last_name,
                               username=user.username, language_code=user.language_code, created_at=datetime.utcnow(),
                               last_active=datetime.utcnow())
                session.add(db_user)
            else:
                db_user.last_active = datetime.utcnow()
                db_user.first_name = user.first_name
                db_user.last_name = user.last_name
                db_user.username = user.username

            await session.commit()
            return db_user

    @staticmethod
    async def update_consultation_count(user_id: int):
        async with async_session() as session:
            user = await DatabaseManager.get_user(user_id)
            if user:
                user.consultation_count += 1
                user.daily_consultation_count += 1
                user.last_consultation_date = datetime.utcnow()
                await session.commit()

    @staticmethod
    async def save_consultation(user_id: int, category: str, question: str, response: str) -> Consultation:
        async with async_session() as session:
            consultation = Consultation(user_id=user_id, category=category, content=question, response=response,
                                        created_at=datetime.utcnow())
            session.add(consultation)
            await session.commit()
            return consultation

    @staticmethod
    async def update_feedback(consultation_id: int, score: int, feedback_text: Optional[str] = None):
        async with async_session() as session:
            consultation = await session.get(Consultation, consultation_id)
            if consultation:
                consultation.feedback_score = score
                consultation.feedback_text = feedback_text
                consultation.is_resolved = True
                consultation.resolved_at = datetime.utcnow()
                await session.commit()


# Statistics manager
class StatisticsManager:
    @staticmethod
    async def get_category_stats(user_id: int) -> List[Any]:
        async with async_session() as session:
            query = (
            select(Consultation.category, func.count(Consultation.id)).where(Consultation.user_id == user_id).group_by(
                Consultation.category).order_by(func.count(Consultation.id).desc()))
            result = await session.execute(query)
            return result.all()

    @staticmethod
    async def get_weekly_activity(user_id: int) -> Dict[str, int]:
        async with async_session() as session:
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            query = (select(func.date(Consultation.created_at), func.count(Consultation.id)).where(
                and_(Consultation.user_id == user_id, Consultation.created_at >= week_ago)).group_by(
                func.date(Consultation.created_at)).order_by(func.date(Consultation.created_at)))
            result = await session.execute(query)
            daily_counts = {str(date): count for date, count in result}
            week_days = {}
            for i in range(7):
                date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
                week_days[date] = daily_counts.get(date, 0)
            return week_days

    @staticmethod
    async def get_feedback_distribution(user_id: int) -> Dict[int, int]:
        async with async_session() as session:
            query = (select(Consultation.feedback_score, func.count(Consultation.id)).where(
                and_(Consultation.user_id == user_id, Consultation.feedback_score.isnot(None))).group_by(
                Consultation.feedback_score).order_by(Consultation.feedback_score))
            result = await session.execute(query)
            return {score: count for score, count in result}

    @staticmethod
    async def get_recent_consultations(user_id: int, limit: int = 5) -> List[Consultation]:
        async with async_session() as session:
            query = (select(Consultation).where(Consultation.user_id == user_id).order_by(
                Consultation.created_at.desc()).limit(limit))
            result = await session.execute(query)
            return result.scalars().all()


# Keyboard functions
def get_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardBuilder()
    keyboard.add(KeyboardButton(text="🩺 Savol berish"))
    keyboard.add(KeyboardButton(text="📊 Statistika"))
    keyboard.add(KeyboardButton(text="ℹ️ Ma'lumot"))
    keyboard.adjust(2)
    return keyboard.as_markup(resize_keyboard=True)


def get_contact_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardBuilder()
    keyboard.add(KeyboardButton(text="📱 Telefon raqamni yuborish", request_contact=True))
    return keyboard.as_markup(resize_keyboard=True)


def get_categories_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardBuilder()
    keyboard.add(KeyboardButton(text="👨‍⚕️ Umumiy maslahat"))
    keyboard.add(KeyboardButton(text="💊 Dori-darmonlar"))
    keyboard.add(KeyboardButton(text="🏥 Kasalxonalar"))
    keyboard.add(KeyboardButton(text="👨‍⚕️ Mutaxassislar"))
    keyboard.add(KeyboardButton(text="🔙 Orqaga"))
    keyboard.adjust(2)
    return keyboard.as_markup(resize_keyboard=True)


# Message handlers
@dp.message(Command("start"))
async def start_command(message: types.Message, state: FSMContext):
    try:
        user_id = message.from_user.id
        await DatabaseManager.add_or_update_user(message.from_user)
        welcome_text = (f"👋 Assalomu alaykum, {message.from_user.first_name}!\n\n"
                        "🏥 Doctor AI - sizning shaxsiy tibbiy maslahatchi botingizga "
                        "xush kelibsiz.\n\n"
                        "🤖 Bot imkoniyatlari:\n"
                        "• Tibbiy maslahatlar\n"
                        "• Dori-darmonlar haqida ma'lumot\n"
                        "• Kasalxonalar va shifokorlar haqida ma'lumot\n\n"
                        "⚠️ Eslatma: Bot bergan maslahatlar faqat umumiy xarakterga ega.")
        await state.clear()
        user = await DatabaseManager.get_user(user_id)
        if user and user.phone_number:
            await message.answer(welcome_text, reply_markup=get_main_keyboard())
        else:
            await message.answer(f"{welcome_text}\n\n📱 Botdan foydalanish uchun telefon raqamingizni yuboring.",
                                 reply_markup=get_contact_keyboard())
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await message.answer("Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")


@dp.message(F.contact)
async def handle_contact(message: types.Message):
    try:
        user_id = message.from_user.id
        async with async_session() as session:
            user = await DatabaseManager.get_user(user_id)
            if user:
                user.phone_number = message.contact.phone_number
                await session.commit()
        await message.answer("✅ Raqamingiz muvaffaqiyatli saqlandi!\n"
                             "Endi botdan to'liq foydalanishingiz mumkin.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in handle_contact: {e}")
        await message.answer("Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")


@dp.message(F.text == "🩺 Savol berish")
async def ask_question(message: types.Message, state: FSMContext):
    try:
        user_id = message.from_user.id
        user = await DatabaseManager.get_user(user_id)
        if user.daily_consultation_count >= MAX_DAILY_CONSULTATIONS:
            await message.answer("⚠️ Siz bugun ko'p savol berdingiz. Iltimos, ertaga qayta urinib ko'ring.")
            return

        await state.set_state(ConsultationStates.category_selection)
        await message.answer("🏥 Qaysi yo'nalishda maslahat olmoqchisiz?", reply_markup=get_categories_keyboard())
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        await message.answer("Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")


@dp.message(ConsultationStates.category_selection)
async def handle_category_selection(message: types.Message, state: FSMContext):
    categories = {"👨‍⚕️ Umumiy maslahat": "general", "💊 Dori-darmonlar": "medicine", "🏥 Kasalxonalar": "hospitals",
                  "👨‍⚕️ Mutaxassislar": "specialists"}

    if message.text == "🔙 Orqaga":
        await state.clear()
        await message.answer("Asosiy menyuga qaytdingiz.", reply_markup=get_main_keyboard())
        return

    if message.text not in categories:
        await message.answer("❌ Noto'g'ri kategoriya. Iltimos, quyidagi tugmalardan birini tanlang.",
                             reply_markup=get_categories_keyboard())
        return

    selected_category = categories[message.text]
    await state.update_data(category=selected_category, category_name=message.text, conversation_history=[])
    await state.set_state(ConsultationStates.conversation)

    keyboard = ReplyKeyboardBuilder()
    keyboard.add(KeyboardButton(text="🔙 Asosiy menyu"))
    keyboard.add(KeyboardButton(text="🔄 Kategoriyani o'zgartirish"))
    keyboard.adjust(1)

    await message.answer(f"✅ <b>{message.text}</b> bo'yicha savol-javob sessiyasi boshlandi.\n\n"
                         "✍️ Savolingizni yozing:\n\n"
                         "📝 Eslatma: Siz asosiy menyuga qaytmaguncha yoki kategoriyani "
                         "o'zgartirmaguncha shu mavzu bo'yicha savollar berishingiz mumkin.",
                         reply_markup=keyboard.as_markup(resize_keyboard=True), parse_mode=ParseMode.HTML)


@dp.message(ConsultationStates.conversation)
async def handle_conversation(message: types.Message, state: FSMContext):
    try:
        if message.text == "🔙 Asosiy menyu":
            await state.clear()
            await message.answer("Asosiy menyuga qaytdingiz.", reply_markup=get_main_keyboard())
            return

        if message.text == "🔄 Kategoriyani o'zgartirish":
            await state.set_state(ConsultationStates.category_selection)
            await message.answer("🏥 Yangi kategoriyani tanlang:", reply_markup=get_categories_keyboard())
            return

        state_data = await state.get_data()
        category = state_data.get('category')
        category_name = state_data.get('category_name')
        conversation_history = state_data.get('conversation_history', [])

        if not category:
            await state.clear()
            await message.answer("Xatolik yuz berdi. Iltimos, qaytadan boshlang.", reply_markup=get_main_keyboard())
            return

        await message.chat.do(ChatAction.TYPING)

        context = f"Kategoriya: {category_name}\n\nOldingi savol-javoblar:\n"
        for qa in conversation_history[-3:]:
            context += f"Savol: {qa['question']}\nJavob: {qa['answer']}\n\n"
        context += f"Yangi savol: {message.text}"

        response = await ai_generator.generate_response(category, context)

        conversation_history.append({'question': message.text, 'answer': response})
        await state.update_data(conversation_history=conversation_history)

        consultation = await DatabaseManager.save_consultation(message.from_user.id, category, message.text, response)
        await DatabaseManager.update_consultation_count(message.from_user.id)

        keyboard = ReplyKeyboardBuilder()
        keyboard.add(KeyboardButton(text="🔙 Asosiy menyu"))
        keyboard.add(KeyboardButton(text="🔄 Kategoriyani o'zgartirish"))
        keyboard.adjust(1)

        await message.answer(response, reply_markup=keyboard.as_markup(resize_keyboard=True))

        feedback_keyboard = InlineKeyboardBuilder()
        for i in range(1, 6):
            feedback_keyboard.add(InlineKeyboardButton(text=f"{'⭐' * i}", callback_data=f"rate_{consultation.id}_{i}"))
        feedback_keyboard.adjust(5)

        await message.answer("Javobdan qanchalik qoniqding? (1-5 yulduz):", reply_markup=feedback_keyboard.as_markup())

    except Exception as e:
        logger.error(f"Error in handle_conversation: {e}")
        await message.answer("Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.", reply_markup=get_main_keyboard())
        await state.clear()


@dp.callback_query(lambda c: c.data.startswith('rate_'))
async def process_feedback(callback_query: types.CallbackQuery):
    try:
        _, consultation_id, score = callback_query.data.split('_')
        await DatabaseManager.update_feedback(int(consultation_id), int(score))
        await callback_query.message.edit_text(f"✅ Rahmat! Sizning bahoyingiz: {'⭐' * int(score)}")
    except Exception as e:
        logger.error(f"Error in process_feedback: {e}")
        await callback_query.message.edit_text("Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")


@dp.message(F.text == "📊 Statistika")
async def show_statistics(message: types.Message):
    try:
        user = await DatabaseManager.get_user(message.from_user.id)
        if not user:
            await message.answer("Foydalanuvchi ma'lumotlari topilmadi.")
            return

        category_stats = await StatisticsManager.get_category_stats(user.user_id)
        weekly_activity = await StatisticsManager.get_weekly_activity(user.user_id)
        feedback_dist = await StatisticsManager.get_feedback_distribution(user.user_id)
        recent_consultations = await StatisticsManager.get_recent_consultations(user.user_id)

        stats_message = ["📊 <b>Statistika</b>\n", f"👤 Foydalanuvchi: {user.first_name}",
                         f"📅 Ro'yxatdan o'tgan sana: {user.created_at.strftime('%Y-%m-%d')}",
                         f"💬 Jami maslahatlar: {user.consultation_count}",
                         f"📈 Bugungi maslahatlar: {user.daily_consultation_count}",
                         f"⭐ O'rtacha baho: {user.feedback_score:.1f}/5.0\n"]

        if category_stats:
            stats_message.append("📊 <b>Kategoriyalar bo'yicha statistika:</b>")
            for category, count in category_stats:
                stats_message.append(f"- {category}: {count} ta")
            stats_message.append("")

        stats_message.append("📅 <b>Haftalik faollik:</b>")
        for date, count in weekly_activity.items():
            weekday = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
            stats_message.append(f"- {weekday}: {count} ta maslahat")
        stats_message.append("")

        if feedback_dist:
            stats_message.append("⭐ <b>Baholar taqsimoti:</b>")
            total_feedback = sum(feedback_dist.values())
            for score, count in feedback_dist.items():
                percentage = (count / total_feedback) * 100
                stars = "⭐" * score
                stats_message.append(f"{stars}: {count} ta ({percentage:.1f}%)")
            stats_message.append("")

        if recent_consultations:
            stats_message.append("🕐 <b>Oxirgi maslahatlar:</b>")
            for cons in recent_consultations:
                date = cons.created_at.strftime("%Y-%m-%d %H:%M")
                stats_message.append(f"- {date}: {cons.category}")

        keyboard = InlineKeyboardBuilder()
        keyboard.add(InlineKeyboardButton(text="📊 Diagrammalar", callback_data="show_charts"))
        keyboard.add(InlineKeyboardButton(text="📥 Eksport (CSV)", callback_data="export_stats"))
        keyboard.adjust(2)

        await message.answer("\n".join(stats_message), reply_markup=keyboard.as_markup(), parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in show_statistics: {e}")
        await message.answer("Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")


@dp.callback_query(lambda c: c.data == "show_charts")
async def show_charts(callback_query: types.CallbackQuery):
    try:
        user_id = callback_query.from_user.id
        category_stats = await StatisticsManager.get_category_stats(user_id)
        weekly_activity = await StatisticsManager.get_weekly_activity(user_id)
        feedback_dist = await StatisticsManager.get_feedback_distribution(user_id)

        # Generate charts using HTML and JavaScript
        html_content = f"""
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div id="category_chart" style="width:100%;height:300px;"></div>
            <div id="weekly_chart" style="width:100%;height:300px;"></div>
            <div id="feedback_chart" style="width:100%;height:300px;"></div>
            <script>
                var categoryData = [{{"values": {[count for _, count in category_stats]}, "labels": {[str(cat) for cat, _ in category_stats]}, "type": "pie"}}];
                var categoryLayout = {{title: "Kategoriyalar bo'yicha statistika"}};
                Plotly.newPlot('category_chart', categoryData, categoryLayout);

                var weeklyData = [{{
                    x: {list(weekly_activity.keys())},
                    y: {list(weekly_activity.values())},
                    type: 'bar'
                }}];
                var weeklyLayout = {{title: 'Haftalik faollik', xaxis: {{title: 'Kunlar'}}, yaxis: {{title: 'Maslahatlar soni'}}}};
                Plotly.newPlot('weekly_chart', weeklyData, weeklyLayout);

                var feedbackData = [{{
                    x: {list(feedback_dist.keys())},
                    y: {list(feedback_dist.values())},
                    type: 'bar'
                }}];
                var feedbackLayout = {{title: 'Baholar taqsimoti', xaxis: {{title: 'Baho'}}, yaxis: {{title: 'Soni'}}}};
                Plotly.newPlot('feedback_chart', feedbackData, feedbackLayout);
            </script>
        </body>
        </html>
        """

        with open("charts.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        await callback_query.message.answer_document(types.FSInputFile("charts.html"),
                                                     caption="📊 Statistika diagrammalari")
        os.remove("charts.html")

    except Exception as e:
        logger.error(f"Error in show_charts: {e}")
        await callback_query.message.answer("Diagrammalarni ko'rsatishda xatolik yuz berdi.")


@dp.callback_query(lambda c: c.data == "export_stats")
async def export_statistics(callback_query: types.CallbackQuery):
    try:
        user_id = callback_query.from_user.id
        consultations = await StatisticsManager.get_recent_consultations(user_id, limit=1000)

        csv_content = "Sana,Kategoriya,Savol,Javob,Baho\n"
        for cons in consultations:
            csv_content += f"{cons.created_at},{cons.category},\"{cons.content}\",\"{cons.response}\",{cons.feedback_score or ''}\n"

        await callback_query.message.answer_document(
            types.BufferedInputFile(csv_content.encode(), filename="statistika.csv"),
            caption="📊 Statistika ma'lumotlari CSV formatida")

    except Exception as e:
        logger.error(f"Error in export_statistics: {e}")
        await callback_query.message.answer("Statistikani eksport qilishda xatolik yuz berdi.")


@dp.message(F.text == "ℹ️ Ma'lumot")
async def show_info(message: types.Message):
    info_text = ("ℹ️ Doctor AI Bot haqida\n\n"
                 "🤖 Bu bot sizga umumiy tibbiy maslahat berish uchun yaratilgan. "
                 "Bot orqali dori-darmonlar, shifokorlar va kasalxonalar haqida "
                 "ma'lumot olishingiz mumkin.\n\n"
                 "⚠️ Eslatma: Bot orqali berilgan ma'lumotlar faqat maslahat uchun bo'lib, "
                 "aniq tashxis va davolanish uchun shifokorga murojaat qilishingiz kerak.")
    await message.answer(info_text, reply_markup=get_main_keyboard())


# Main execution
async def main():
    logging.info("Starting the bot...")
    try:
        await DatabaseManager.init_db()
        logging.info("Database initialized successfully")
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Error during bot startup: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
