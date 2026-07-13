/**
 * Endless shuffled walk over the bundled sample digits for attract mode:
 * every sample appears once per cycle, and a reshuffle never plays the
 * same digit twice in a row across the cycle boundary.
 */
export interface AttractScheduler {
  next(): number;
}

export function createAttractScheduler(
  count: number,
  rand: () => number = Math.random
): AttractScheduler {
  if (count <= 0) throw new Error("scheduler needs at least one sample");
  let order = shuffle(count, rand);
  let cursor = 0;
  return {
    next() {
      if (cursor === order.length) {
        const last = order[order.length - 1];
        do {
          order = shuffle(count, rand);
        } while (count > 1 && order[0] === last);
        cursor = 0;
      }
      return order[cursor++];
    },
  };
}

function shuffle(count: number, rand: () => number): number[] {
  const order = Array.from({ length: count }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}
