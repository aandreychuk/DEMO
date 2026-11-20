export type Language = 'en' | 'ru';

const RU: Record<string, string> = {
  'MAPF Operations Demo': 'Демонстрация MAPF',
  'DMM Communication Maze': 'Лабиринт коммуникации DMM',
  'Multi-agent pathfinding demo': 'Демонстрация многоагентного планирования',
  'Demo views': 'Разделы демо',
  'Warehouse': 'Склад',
  'EPIBT warehouse': 'Склад EPIBT',
  'Warehouse map': 'Карта склада',
  'Warehouse map with oriented robots': 'Карта склада с направлениями роботов',
  'Import JSON': 'Загрузить JSON',
  'Export JSON': 'Выгрузить JSON',
  'Pickup bay': 'Зона забора',
  'Delivery bay': 'Зона доставки',
  'Robot': 'Робот',
  'With cargo': 'С грузом',
  'Loading EPIBT…': 'Загрузка EPIBT…',
  'Completed deliveries': 'Доставок',
  'Planning, ms': 'Планирование, мс',
  'Heading': 'Ориентация',
  'Task': 'Задача',
  'Current goal': 'Текущая цель',
  'Last action': 'Последнее действие',
  'Planning priority': 'Приоритет при планировании',
  'Last EPIBT operation': 'Последний план EPIBT',
  'Previous robot': 'Предыдущий робот',
  'Next robot': 'Следующий робот',
  'Planning horizon': 'Горизонт планирования',
  'Changing the planning horizon…': 'Смена горизонта планирования…',
  'Could not change the planning horizon.': 'Не удалось изменить горизонт планирования.',
  'Cells travelled': 'Пройдено клеток',
  'Turns': 'Поворотов',
  'Wait ticks': 'Тактов ожидания',
  'Fleet': 'Роботы',
  'Single step': 'Один шаг',
  'Forward': 'Вперёд',
  'Turn right': 'Поворот вправо',
  'Turn left': 'Поворот влево',
  'Delivery': 'Доставить',
  'East': 'восток',
  'South': 'юг',
  'West': 'запад',
  'North': 'север',
  'Robot {id}': 'Робот {id}',
  'Running': 'В работе',
  'Paused': 'Пауза',
  'EPIBT stopped': 'EPIBT остановлен',
  'Could not start the EPIBT worker.': 'Не удалось запустить EPIBT.',
  'The JSON file is larger than 1 MB.': 'Файл JSON больше 1 МБ.',
  'The file does not contain valid JSON.': 'В файле некорректный JSON.',
  'Could not read the JSON file.': 'Не удалось прочитать файл JSON.',
  'Expected an EPIBT warehouse JSON file.': 'Выберите JSON-файл конфигурации склада EPIBT.',
  'The map must have 2–40 rows.': 'На карте должно быть от 2 до 40 строк.',
  'The map must have 2–48 columns and use only #, ., S and D.': 'На карте должно быть от 2 до 48 столбцов. Допустимые символы — #, ., S и D.',
  'The warehouse must have 1–200 robots.': 'Укажите от 1 до 200 роботов.',
  'Place at least one pickup and one delivery bay.': 'Добавьте хотя бы одну зону забора и одну зону доставки.',
  'Connect all pickup and delivery bays by aisles.': 'Соедините все зоны забора и доставки проходами.',
  'There are too few reachable cells for this many robots.': 'Для такого числа роботов не хватает доступных клеток.',
  'Edit map': 'Редактировать карту',
  'Map editor': 'Редактор карты',
  '↶ Undo edit': '↶ Отменить правку',
  'Warehouse preset': 'Готовая конфигурация',
  'Custom layout': 'Своя карта',
  'Number of robots': 'Число роботов',
  'Rack rows': 'Стеллажные ряды',
  'Central junction': 'Центральный перекрёсток',
  'Narrow aisles': 'Узкие проходы',
  'Staggered gates': 'Смещённые ворота',
  'Switchback lanes': 'Серпантин',
  'Nested loops': 'Кольцевые маршруты',
  'Paint cells': 'Тип клетки',
  'Obstacle': 'Препятствие',
  'Click or drag on the map to paint.': 'Нажмите на клетку или проведите по карте.',
  'Editable warehouse map. Use arrow keys and Space to paint.': 'Карта склада. Стрелки перемещают курсор, пробел меняет клетку.',
  'EPIBT is not ready.': 'EPIBT ещё не готов.',
  'EPIBT returned an invalid move.': 'EPIBT вернул недопустимый ход.',
  'Communication maze': 'Лабиринт коммуникации',
  'Demo source': 'Источник демо',
  'QR links': 'QR-ссылки',
  'Demo links': 'Ссылки на демо',
  'Explore DMM': 'Подробнее о DMM',
  'Explore EPIBT': 'Подробнее об EPIBT',
  'Code': 'Код',
  'Implementation': 'Реализация',
  'Live demo': 'Демо',
  'Paper': 'Статья',
  'Open the method implementation': 'Открыть реализацию метода',
  'Open the live demo': 'Открыть демо',
  'Open the paper': 'Открыть статью',
  'QR code for the DMM implementation': 'QR-код реализации DMM',
  'QR code for the EPIBT implementation': 'QR-код реализации EPIBT',
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
  'Pallet': 'Палета',
  'Break agent': 'Сломать агента',
  'Take control': 'Управлять',
  'Release control': 'Вернуть управление DMM',
  'Move the robot with WASD, the left stick, or the D-pad. Esc returns control to DMM.': 'WASD, левый стик или крестовина — движение робота. Esc — вернуть управление DMM.',
  'Live telemetry': 'Телеметрия',
  'Throughput': 'Пропускная способность',
  'Tasks per step': 'задач/шаг',
  'Environment steps': 'Шаги среды',
  'Inference, ms': 'Инференс, мс',
  'Agents': 'Агенты',
  'Step': 'Шаг',
  'Failed': 'Сломаны',
  'Recovery': 'На ремонте',
  'In recovery': 'Восстановление',
  'Recovery unit': 'Эвакуатор',
  'Waiting': 'Ожидание',
  'Handoff': 'Передача',
  'Warehouse configuration': 'Конфигурация склада',
  'Pallet layout': 'Расстановка палет',
  'Click a cell to add or remove a pallet. Drag to paint. Pan with WASD or the left stick. Zoom with the wheel or L2/R2.': 'Нажмите на клетку, чтобы добавить или убрать палету. Перетаскивайте для рисования. WASD или левый стик — движение камеры, колесо или L2/R2 — масштаб.',
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
  'Gamepad controls': 'Управление геймпадом',
  'Select at reticle': 'Выбрать в прицеле',
  'Switch robot': 'Сменить робота',
  'Toggle manual control': 'Ручное управление',
  'Clear selection': 'Снять выбор',
  'Pause / resume': 'Пауза / продолжить',
  'Stop simulation': 'Остановить симуляцию',
  'Open layout editor': 'Открыть редактор склада',
  'Simulation speed': 'Скорость симуляции',
  'Left stick to move, right stick to orbit, L2/R2 to zoom': 'Левый стик — движение, правый — поворот, L2/R2 — масштаб',
  'Hold L2 + R2 in manual mode to change speed with ↑ / ↓.': 'В ручном режиме зажми L2 + R2, чтобы менять скорость кнопками ↑ / ↓.',
  'Layout editor gamepad controls': 'Управление редактором с геймпада',
  'Add or remove pallet': 'Добавить или убрать палету',
  'Undo edit': 'Отменить правку',
  'Change robot count by 1': 'Изменить число роботов на 1',
  'Change robot count by 10': 'Изменить число роботов на 10',
  'Left stick to aim at a cell, L2/R2 to zoom': 'Левый стик — навести прицел на клетку, L2/R2 — масштаб',
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
  ' · at {x}, {z}': ' · в клетке {x}, {z}',
  'Planning {count}': 'Планирование: {count}',
  'The local runtime is not connected.': 'Локальный движок не подключён.',
  'Saving the layout and rebuilding the task queue…': 'Сохранение расстановки и перестройка очереди задач…',
  'Add at least one pallet to generate delivery tasks.': 'Добавьте хотя бы одну палету для создания задач.',
  'The loading and unloading sides are disconnected.': 'Зоны загрузки и разгрузки не соединены.',
  'Heading to robot': 'Едет к роботу',
  'Taking robot to repair': 'Везёт на ремонт',
  'Returning to depot': 'Возвращается на базу',
  'Lifting robot': 'Поднимает робота',
  'Active': 'Работает',
  'One action, four rounds of negotiation': 'Одно действие, четыре раунда коммуникации',
  'Loading model': 'Загрузка модели',
  'Mini maze and communicating agents': 'Мини-лабиринт и обменивающиеся сообщениями агенты',
  'Live policy trace': 'Работа модели по раундам',
  'The decision before the move': 'Решение до движения',
  'Arrows show the agents’ intents. Lines show who can exchange messages. Agents move after the fourth round.': 'Стрелки показывают намерения агентов. Линии показывают, кто может обмениваться сообщениями. Агенты двигаются после четвёртого раунда.',
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
  'Votes by round': 'Голоса по раундам',
  'Votes and final intent scores for the selected agent': 'Голоса и итоговые оценки намерений выбранного агента',
  'Final score': 'Оценка',
  'Final intent score': 'Итоговая оценка намерения',
  'Final intent score for {action}, {score}': 'Итоговая оценка действия «{action}» — {score}',
  'Chosen action': 'Выбранный ход',
  'Message sources': 'Источники сообщений',
  '0 agents': '0 агентов',
  'All agents': 'Все агенты',
  'Tap to inspect': 'Нажмите для просмотра',
  'Votes update the intent scores. After the fourth round, the planner uses the scores to choose collision-free moves.': 'Голоса меняют оценки намерений. После четвёртого раунда планировщик использует эти оценки, чтобы выбрать ходы без столкновений.',
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
  'Up': 'Вверх',
  'Down': 'Вниз',
  'Left': 'Влево',
  'Right': 'Вправо',
  'No agents in range': 'В радиусе связи нет агентов',
  'Inspect agent {id}': 'Посмотреть агента {id}',
  'Goal ({x}, {y}) reached.': 'Цель ({x}, {y}) достигнута.',
  'Goal ({x}, {y}).': 'Цель ({x}, {y}).',
  'Round {round}': 'Раунд {round}',
  'Round {round} vote: {action}': 'Голос раунда {round}: {action}',
  'Agent {id}: {state}': 'Агент {id}: {state}',
  'before communication': 'до коммуникации',
  'Communication round {round} of {total}': 'Раунд коммуникации {round} из {total}',
  'Step {step}': 'Шаг {step}',
  'Agent {id}: goal reached': 'Агент {id}: цель достигнута',
  'Agent {id}: awaiting communication': 'Агент {id}: ожидает коммуникацию',
  'Agent {id}, intent {action} ({value})': 'Агент {id}, намерение — {action} ({value})',
  'Intents begin in the next round': 'Намерения появятся в следующем раунде',
  'DMM-0.8M on {backend}': 'DMM-0.8M на {backend}',
  'The model supports up to {count} agents here.': 'Здесь модель поддерживает до {count} агентов.',
  "Now set agent {id}'s goal.": 'Теперь задайте цель агенту {id}.',
  'Loading the selected maze scenario…': 'Загрузка выбранного сценария…',
  'Loading DMM-0.8M trace on WASM…': 'Загрузка трассировки DMM-0.8M на WASM…',
  'Loading DMM-0.8M communication trace…': 'Загрузка трассировки коммуникации DMM-0.8M…',
  'Computing the first communication trace…': 'Вычисление первой трассировки коммуникации…',
  'The model returned invalid intent scores.': 'Модель вернула некорректные оценки намерений.',
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

const COUNT_FORMS = {
  agents: {
    en: ['agent', 'agents'],
    ru: ['агент', 'агента', 'агентов'],
  },
  stepsPerSecond: {
    en: ['step/s', 'steps/s'],
    ru: ['шаг/с', 'шага/с', 'шагов/с'],
  },
  cellsTravelled: {
    en: ['cell travelled', 'cells travelled'],
    ru: ['клетка пройдена', 'клетки пройдены', 'клеток пройдено'],
  },
  waitTicks: {
    en: ['wait tick', 'wait ticks'],
    ru: ['такт ожидания', 'такта ожидания', 'тактов ожидания'],
  },
  unloads: {
    en: ['unload', 'unloads'],
    ru: ['разгрузка', 'разгрузки', 'разгрузок'],
  },
  tasksDone: {
    en: ['task done', 'tasks done'],
    ru: ['задача выполнена', 'задачи выполнены', 'задач выполнено'],
  },
  intentsFormed: {
    en: ['intent formed', 'intents formed'],
    ru: ['намерение сформировано', 'намерения сформированы', 'намерений сформировано'],
  },
  intentsChanged: {
    en: ['changed intent this round', 'changed intents this round'],
    ru: ['намерение изменилось в этом раунде', 'намерения изменились в этом раунде', 'намерений изменилось в этом раунде'],
  },
  robotStartCells: {
    en: ['robot start cell', 'robot start cells'],
    ru: ['стартовая клетка для роботов', 'стартовые клетки для роботов', 'стартовых клеток для роботов'],
  },
  palletCells: {
    en: ['pallet cell', 'pallet cells'],
    ru: ['клетка с палетой', 'клетки с палетами', 'клеток с палетами'],
  },
} as const;

export type CountLabel = keyof typeof COUNT_FORMS;
const russianPluralRules = new Intl.PluralRules('ru-RU');

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
export function countLabel(count: number, label: CountLabel): string {
  const forms = COUNT_FORMS[label];
  if (language === 'en') return forms.en[count === 1 ? 0 : 1];
  const category = russianPluralRules.select(count);
  return forms.ru[category === 'one' ? 0 : category === 'few' ? 1 : 2];
}
export function formatCount(count: number, label: CountLabel): string {
  return `${formatNumber(count)} ${countLabel(count, label)}`;
}
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
  const startCells = /^The map has only (\d+) robot start cells\.$/.exec(english);
  if (startCells) return `На карте только ${formatCount(Number(startCells[1]), 'robotStartCells')}.`;
  const blockedPallets = /^(\d+) pallet cells have no loaded route to an aisle\.?$/.exec(english);
  if (blockedPallets) return `${formatCount(Number(blockedPallets[1]), 'palletCells')} без маршрута к проходу для робота с грузом.`;
  const patterns: Array<[RegExp, string]> = [
    [/^Enter a whole number of agents from 1 to (\d+)\.$/, 'Введите целое число агентов от 1 до $1.'],
    [/^Choose 1–(\d+) agents\.$/, 'Выберите число агентов от 1 до $1.'],
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
