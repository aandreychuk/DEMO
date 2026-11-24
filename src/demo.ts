import './demo-shell.css';
import './demo-header.css';
import './style.css';
import './maze.css';
import { getLanguage, initI18n, tr } from './i18n';
import type { DemoView } from './demo-view';
import warehouseMarkup from './views/warehouse.html?raw';
import mazeMarkup from './views/maze.html?raw';

const host = document.querySelector<HTMLElement>('#demo-views')!;
host.innerHTML = warehouseMarkup + mazeMarkup;

const panels = {
  warehouse: host.querySelector<HTMLElement>('[data-demo-panel="warehouse"]')!,
  maze: host.querySelector<HTMLElement>('[data-demo-panel="maze"]')!,
};
const links = host.querySelectorAll<HTMLAnchorElement>('a[data-demo-link]');
const titles = { warehouse: 'Interactive warehouse', maze: 'DMM communication' };
let currentView: DemoView = document.body.dataset.demoView === 'maze' ? 'maze' : 'warehouse';
initI18n();

function renderTitle(): void {
  document.title = tr(titles[currentView]);
}

function renderView(view: DemoView, focusTab = false): void {
  currentView = view;
  document.body.dataset.demoView = view;
  panels.warehouse.hidden = view !== 'warehouse';
  panels.maze.hidden = view !== 'maze';
  for (const link of links) {
    if (link.dataset.demoLink === view) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  renderTitle();
  window.dispatchEvent(new Event('demo:viewchange'));
  if (focusTab) panels[view].querySelector<HTMLAnchorElement>('a[aria-current="page"]')?.focus({ preventScroll: true });
}

host.addEventListener('click', (event) => {
  if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const link = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[data-demo-link]') : null;
  if (!link) return;
  event.preventDefault();
  const view = link.dataset.demoLink as DemoView;
  if (view === currentView) return;
  history.pushState(null, '', link.href);
  renderView(view, true);
});

window.addEventListener('popstate', () => {
  const url = new URL(location.href);
  if (getLanguage() === 'ru') url.searchParams.set('lang', 'ru');
  else url.searchParams.delete('lang');
  if (url.href !== location.href) history.replaceState(history.state, '', url);
  renderView(url.pathname.endsWith('/maze.html') ? 'maze' : 'warehouse');
});
window.addEventListener('demo:languagechange', renderTitle);

renderView(currentView);
void import('./main');
void import('./maze');
