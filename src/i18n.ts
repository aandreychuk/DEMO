export type Language = 'en' | 'ru';

const RU: Record<string, string> = {
  'MAPF Operations Demo': 'Демонстрация MAPF',
  'DMM Communication Maze': 'Лабиринт коммуникации DMM',
  'Multi-agent pathfinding demo': 'Демонстрация многоагентного планирования',
  'Demo views': 'Разделы демо',
  'Warehouse': 'Склад',
  'Communication maze ↗': 'Лабиринт коммуникации ↗',
  'Communication maze': 'Лабиринт коммуникации',
  'Demo source': 'Источник демо',
  'QR links': 'QR-ссылки',
  'Demo links': 'Ссылки на демо',
  'Explore DMM': 'Подробнее о DMM',
  'Implementation': 'Реализация',
  'Live demo': 'Демо',
  'Paper': 'Статья',
  'Open the method implementation': 'Открыть реализацию метода',
  'Open the live demo': 'Открыть демо',
  'Open the paper': 'Открыть статью',
  'QR code for the DMM implementation': 'QR-код реализации DMM',
  'QR code for the live demo': 'QR-код демо',
  'QR code for the paper': 'QR-код статьи',
  '3D MAPF simulation': 'Трёхмерная симуляция MAPF',
  'About this demo': 'О демонстрации',
  'What to watch': 'На что смотреть',
  'Robots carry pallets to unloading arms, refill empty ones, then return them to storage.': 'Роботы везут палеты к разгрузочным манипуляторам, пополняют опустевшие палеты и возвращают их на склад.',
  'Select a robot to inspect its task, steer it, or trigger a breakdown. Use Layout to edit pallets and agent count.': 'Выберите робота, чтобы посмотреть его задачу, управлять им или сломать его. В редакторе можно менять палеты и число агентов.',
  'Agent inspector': 'Информация об агенте',
  'Clear agent selection': 'Снять выделение агента',
  'Select an agent': 'Выберите агента',
  'Click a robot or its carried pallet': 'Нажмите на робота или палету, которую он везёт',
  'Agent 000': 'Агент 000',
  'Empty': 'Без груза',
  'Loaded': 'С грузом',
  'Task progress': 'Ход выполнения задачи',
  'Pickup': 'Забрать',
  'Unload': 'Разгрузить',
  'Reload': 'Пополнить',
  'Return': 'Вернуть',
  'Position': 'Позиция',
  'Current task': 'Текущая задача',
  'Pallet': 'Палета',
  'Task age': 'Время задачи',
  'Reload if empty': 'Пополнить, если пуста',
  'Cells travelled': 'Клеток пройдено',
  'Wait ticks': 'Тактов ожидания',
  'Unloads': 'Разгрузок',
  'Tasks done': 'Задач выполнено',
  'Break agent': 'Сломать агента',
  'Take control': 'Управлять',
  'Release control': 'Вернуть управление DMM',
  'WASD moves along the grid. Esc returns control to DMM.': 'WASD — движение по сетке. Esc — вернуть управление DMM.',
  'Live telemetry': 'Телеметрия',
  'Agents': 'Агенты',
  'Step': 'Шаг',
  'Moves/s': 'Ходов/с',
  'Inference': 'Инференс',
  'Failed': 'Сломаны',
  'Recovery': 'На ремонте',
  'Throughput: tasks/step': 'Пропускная способность: задач/шаг',
  'Recovery unit': 'Эвакуатор',
  'Idle': 'Ожидает',
  'Waiting': 'Ожидание',
  'Handoff': 'Передача',
  'Warehouse configuration': 'Конфигурация склада',
  'Pallet layout': 'Расстановка палет',
  'Click a cell to add or remove a pallet. Drag to paint. Use WASD to move and the wheel to zoom.': 'Нажмите на клетку, чтобы добавить или убрать палету. Перетаскивайте для рисования. WASD — движение камеры, колесо — масштаб.',
  'Pallets': 'Палеты',
  'Density': 'Плотность',
  'Number of agents': 'Число агентов',
  'Number of agents slider': 'Ползунок числа агентов',
  '1000 max': 'макс. 1000',
  'Layout valid': 'Расстановка корректна',
  'Layout blocked': 'Расстановка недопустима',
  'Every pallet has a route to the service aisles.': 'От каждой палеты есть маршрут к служебным проходам.',
  'Aisle': 'Проход',
  'Service zone': 'Служебная зона',
  '↶ Undo': '↶ Отменить',
  'Reset': 'Сбросить',
  'Clear': 'Очистить',
  'Changes are applied to the local simulator.': 'Изменения применяются к локальному симулятору.',
  'Cancel': 'Отмена',
  'Apply & restart': 'Применить и перезапустить',
  'Pause': 'Пауза',
  'Run': 'Продолжить',
  'Stop': 'Стоп',
  'Speed': 'Скорость',
  'Layout': 'Расстановка',
  'Reset camera': 'Сбросить камеру',
  'WASD to move, drag to orbit, scroll to zoom': 'WASD — движение, перетаскивание — поворот, колесо — масштаб',
  'Loading onto tow': 'Погрузка на эвакуатор',
  'Waiting for tow': 'Ожидает эвакуатор',
  'Evacuating': 'Эвакуация',
  'Under repair': 'На ремонте',
  'Recovering pallet': 'Возвращается к палете',
  'Loading pallet': 'Забирает палету',
  'Parking pallet': 'Ставит палету',
  'Batch loading': 'Пополнение партии',
  'Unloading item': 'Разгружает товар',
  'Recovery in progress': 'Идёт восстановление',
  'Breaking…': 'Ломается…',
  'Crate': 'Ящики',
  'Cartons': 'Коробки',
  'Drums': 'Бочки',
  'Connecting locally': 'Подключение к локальному движку',
  'Planning lifelong run': 'Планирование непрерывного сценария',
  'DMM-0.8M simulator': 'Симулятор DMM-0.8M',
  'Demo reconnecting': 'Повторное подключение',
  'Layout editor': 'Редактор расстановки',
  'Simulation stopped': 'Симуляция остановлена',
  'Runtime error': 'Ошибка движка',
  'The layout could not be applied.': 'Не удалось применить расстановку.',
  '{max} max': 'макс. {max}',
  'Agent {id}': 'Агент {id}',
  ', at {x},{z}': ', в клетке {x},{z}',
  '{count} ticks': '{count} тактов',
  'Planning {count}': 'Планирование: {count}',
  ', queue {count}': ', в очереди: {count}',
  'The local runtime is not connected.': 'Локальный движок не подключён.',
  'Saving the layout and rebuilding the task queue…': 'Сохранение расстановки и перестройка очереди задач…',
  'Add at least one pallet to generate delivery tasks.': 'Добавьте хотя бы одну палету для создания задач.',
  'The loading and unloading sides are disconnected.': 'Зоны загрузки и разгрузки не соединены.',
  'To agent': 'К агенту',
  'Transporting': 'Перевозит',
  'Returning': 'Возвращается',
  'Loading': 'Загружает',
  'Active': 'Работает',
  'Demo': 'Демо',
  'One action, four rounds of negotiation': 'Одно действие, четыре раунда коммуникации',
  'Loading model': 'Загрузка модели',
  'Mini maze and communicating agents': 'Мини-лабиринт и обменивающиеся сообщениями агенты',
  'Live policy trace': 'Работа модели по раундам',
  'The decision before the move': 'Решение до движения',
  'Each arrow points to an agent’s strongest z value. Connections show who can exchange messages. The agents move after round four.': 'Стрелка показывает действие с наибольшим z. Линии показывают обмен сообщениями. Агенты двигаются после четвёртого раунда.',
  'Action': 'Действие',
  'Round': 'Раунд',
  '✓ Solved — every agent reached its goal': '✓ Решено — все агенты достигли целей',
  '15 by 11 cell maze': 'Лабиринт 15 × 11 клеток',
  'Wall': 'Стена',
  'Agent goal': 'Цель агента',
  'Communication': 'Коммуникация',
  '0 changed intents this round': '0 изменений намерений в этом раунде',
  'Intent inspector': 'Намерения агента',
  'Agent 01': 'Агент 01',
  'Select an agent to inspect its decision.': 'Выберите агента, чтобы изучить его решение.',
  'Rounds for selected agent': 'Раунды выбранного агента',
  'How z changes': 'Как меняется z',
  'Accumulated action scores': 'Накопленные оценки действий',
  'Votes that update z': 'Голоса, изменяющие z',
  'Four communication rounds': 'Четыре раунда коммуникации',
  "Selected agent's vote in each round": 'Голос выбранного агента в каждом раунде',
  'Message sources': 'Источники сообщений',
  '0 agents': '0 агентов',
  'All agents': 'Все агенты',
  'Tap to inspect': 'Нажмите для просмотра',
  "The line chart shows accumulated z. Each filled cell below marks the action chosen by that round's vote, which pushes z toward that action. The final z drives movement with collision shielding.": 'График показывает накопленное z. Заполненные клетки отмечают голоса раундов, которые изменяют z. Итоговое z определяет движение с защитой от столкновений.',
  'Scenario editor': 'Редактор сценария',
  'Build your maze': 'Создайте лабиринт',
  'Choose a tool, then click cells on the board. Drag to paint walls or erase them. Select an agent to move its start or goal.': 'Выберите инструмент и нажимайте на клетки. Перетаскивайте, чтобы рисовать или стирать стены. Выберите агента, чтобы перенести его старт или цель.',
  'Maze editing tools': 'Инструменты редактора лабиринта',
  'Erase': 'Стереть',
  'Add agent': 'Добавить агента',
  'Move start': 'Перенести старт',
  'Set goal': 'Задать цель',
  'Remove selected agent': 'Удалить выбранного агента',
  'Lifelong': 'Непрерывный режим',
  'Generate a new goal when an agent arrives': 'Создавать новую цель по прибытии агента',
  'Default': 'По умолчанию',
  'Run scenario': 'Запустить сценарий',
  'Communication round player': 'Плеер раундов коммуникации',
  'Previous communication round': 'Предыдущий раунд коммуникации',
  'Next communication round': 'Следующий раунд коммуникации',
  'Before communication': 'До коммуникации',
  'Step 0': 'Шаг 0',
  'Recorded communication round': 'Записанный раунд коммуникации',
  'Playback': 'Воспроизведение',
  'Restart': 'Перезапустить',
  'Edit maze': 'Редактировать лабиринт',
  'Play': 'Запустить',
  'Wait': 'Ожидать',
  'North': 'Север',
  'South': 'Юг',
  'West': 'Запад',
  'East': 'Восток',
  'No agents in range': 'В радиусе связи нет агентов',
  'Inspect agent {id}': 'Посмотреть агента {id}',
  'Selected agent raw z values over four communication rounds': 'Значения z выбранного агента за четыре раунда коммуникации',
  'Goal ({x}, {y}) reached. The agent stays here while the others finish.': 'Цель ({x}, {y}) достигнута. Агент остаётся здесь, пока остальные заканчивают.',
  'Goal ({x}, {y}). Communication has not started.': 'Цель ({x}, {y}). Коммуникация ещё не началась.',
  'Goal ({x}, {y}). Leading z: {action} {z}. Last executed move: {move}.': 'Цель ({x}, {y}). Лидирует z для действия «{action}»: {z}. Последний ход: {move}.',
  'Round {round}: {action}, z {value}': 'Раунд {round}: {action}, z {value}',
  'Round {round}': 'Раунд {round}',
  'Round {round}: {action}': 'Раунд {round}: {action}',
  'Round {round} vote: {action}': 'Голос раунда {round}: {action}',
  'Before the first vote': 'До первого голосования',
  'Agent {id}: {state}': 'Агент {id}: {state}',
  'before communication': 'до коммуникации',
  'Communication round {round} of {total}': 'Раунд коммуникации {round} из {total}',
  'Step {step}': 'Шаг {step}',
  'Agent {id}: goal reached': 'Агент {id}: цель достигнута',
  'Agent {id}: awaiting communication': 'Агент {id}: ожидает коммуникацию',
  'Agent {id}: {action}, z {value}': 'Агент {id}: {action}, z {value}',
  'Intents begin in the next round': 'Намерения появятся в следующем раунде',
  'DMM-0.8M on {backend}': 'DMM-0.8M на {backend}',
  'The model supports up to {count} agents here.': 'Здесь модель поддерживает до {count} агентов.',
  "Now set agent {id}'s goal.": 'Теперь задайте цель агенту {id}.',
  'Loading the selected maze scenario…': 'Загрузка выбранного сценария…',
  'Loading DMM-0.8M trace on WASM…': 'Загрузка трассировки DMM-0.8M на WASM…',
  'Loading DMM-0.8M communication trace…': 'Загрузка трассировки коммуникации DMM-0.8M…',
  'Computing the first communication trace…': 'Вычисление первой трассировки коммуникации…',
  'Maze worker failed': 'Ошибка вычислителя лабиринта',
  'Invalid maze configuration.': 'Некорректная конфигурация лабиринта.',
  'Invalid wall positions.': 'Некорректное расположение стен.',
  'The outer border must stay closed.': 'Внешняя граница должна оставаться закрытой.',
  'Two agents cannot share a start cell.': 'Два агента не могут начинать в одной клетке.',
  'Two agents cannot share a goal cell.': 'Два агента не могут иметь одну и ту же цель.',
  'The outer border stays closed.': 'Внешняя граница остаётся закрытой.',
  'Add an agent first.': 'Сначала добавьте агента.',
  'Move the agent or goal before placing a wall here.': 'Перенесите агента или цель, прежде чем ставить здесь стену.',
  'Choose a free floor cell for the new agent.': 'Выберите свободную клетку для нового агента.',
  'Choose a free floor cell for the start.': 'Выберите свободную клетку для старта.',
  'Choose a floor cell without another goal.': 'Выберите клетку без чужой цели.',
  'Default maze restored. Run the scenario to apply it.': 'Исходный лабиринт восстановлен. Запустите сценарий, чтобы применить его.',
};

function storedLanguage(): Language {
  const fromUrl = new URLSearchParams(location.search).get('lang');
  if (fromUrl === 'ru' || fromUrl === 'en') return fromUrl;
  try { return localStorage.getItem('demo-language') === 'ru' ? 'ru' : 'en'; }
  catch { return 'en'; }
}

let language = storedLanguage();
let initialized = false;
let englishTitle = '';
const textBindings: Array<{ node: Text; english: string }> = [];
const attributeBindings: Array<{ node: Element; name: string; english: string }> = [];

export function getLanguage(): Language { return language; }
export function tr(english: string): string { return language === 'ru' ? RU[english] ?? english : english; }
export function formatNumber(value: number): string { return value.toLocaleString(language === 'ru' ? 'ru-RU' : 'en-US'); }
export function formatFixed(value: number, digits: number): string {
  return value.toLocaleString(language === 'ru' ? 'ru-RU' : 'en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export function fmt(english: string, values: Record<string, string | number>): string {
  return tr(english).replace(/\{(\w+)\}/g, (_, key: string) => {
    const value = values[key];
    return typeof value === 'number' ? formatNumber(value) : String(value ?? '');
  });
}

export function translateMessage(english: string): string {
  if (language === 'en') return english;
  if (RU[english]) return RU[english];
  const patterns: Array<[RegExp, string]> = [
    [/^Enter a whole number of agents from 1 to (\d+)\.$/, 'Введите целое число агентов от 1 до $1.'],
    [/^The map has only (\d+) robot start cells\.$/, 'На карте только $1 стартовых клеток для роботов.'],
    [/^(\d+) pallet cells have no loaded route to an aisle\.$/, 'У $1 клеток с палетами нет маршрута с грузом к проходу.'],
    [/^Choose 1–(\d+) agents\.$/, 'Выберите от 1 до $1 агентов.'],
    [/^Agent (\d+) starts on a wall\.$/, 'Агент $1 начинает на стене.'],
    [/^Choose a goal for agent (\d+)\.$/, 'Задайте цель агенту $1.'],
    [/^Agent (\d+) has a goal on a wall\.$/, 'Цель агента $1 находится на стене.'],
    [/^Agent (\d+) cannot reach its goal through the walls\.$/, 'Агент $1 не может достичь цели из-за стен.'],
  ];
  for (const [pattern, replacement] of patterns) if (pattern.test(english)) return english.replace(pattern, replacement);
  return english;
}

function applyLanguage(): void {
  document.documentElement.lang = language;
  document.title = tr(englishTitle);
  for (const { node, english } of textBindings) {
    if (!node.isConnected) continue;
    const trimmed = english.trim();
    node.data = trimmed ? english.replace(trimmed, tr(trimmed)) : english;
  }
  for (const { node, name, english } of attributeBindings) if (node.isConnected) node.setAttribute(name, tr(english));
  for (const button of document.querySelectorAll<HTMLButtonElement>('.lang-toggle')) {
    button.textContent = language === 'ru' ? 'EN' : 'RU';
    button.title = language === 'ru' ? 'Переключить на английский' : 'Переключить на русский';
    button.setAttribute('aria-label', button.title);
  }
  for (const link of document.querySelectorAll<HTMLAnchorElement>('nav a[href]')) {
    const url = new URL(link.href);
    if (language === 'ru') url.searchParams.set('lang', 'ru');
    else url.searchParams.delete('lang');
    link.href = url.href;
  }
}

export function setLanguage(next: Language): void {
  if (next === language) return;
  language = next;
  try { localStorage.setItem('demo-language', language); } catch { /* Private browsing may disable storage. */ }
  const url = new URL(location.href);
  if (language === 'ru') url.searchParams.set('lang', 'ru');
  else url.searchParams.delete('lang');
  history.replaceState(history.state, '', url);
  applyLanguage();
  window.dispatchEvent(new Event('demo:languagechange'));
}

export function initI18n(): void {
  if (initialized) return;
  initialized = true;
  try { localStorage.setItem('demo-language', language); } catch { /* Private browsing may disable storage. */ }
  englishTitle = document.title;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode as Text;
    if (node.parentElement?.closest('script, style')) continue;
    if (RU[node.data.trim()]) textBindings.push({ node, english: node.data });
  }
  for (const node of document.querySelectorAll('*')) {
    for (const name of ['aria-label', 'title', 'alt', 'placeholder']) {
      const english = node.getAttribute(name);
      if (english && RU[english]) attributeBindings.push({ node, name, english });
    }
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>('.lang-toggle')) {
    button.addEventListener('click', () => setLanguage(language === 'ru' ? 'en' : 'ru'));
  }
  applyLanguage();
}
