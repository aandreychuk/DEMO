export type DemoView = 'warehouse' | 'maze';

export function onViewActivityChange(view: DemoView, update: (active: boolean) => void): void {
  let previous: boolean | undefined;
  const sync = () => {
    const active = document.body.dataset.demoView === view && !document.hidden;
    if (active === previous) return;
    previous = active;
    update(active);
  };
  window.addEventListener('demo:viewchange', sync);
  document.addEventListener('visibilitychange', sync);
  sync();
}
