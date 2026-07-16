import { Component, Fragment, type ReactNode } from "react";

/** How long a crash stays dark before the tree is remounted and retried. */
export const REMOUNT_DELAY_MS = 5000;

interface Props {
  children: ReactNode;
}

interface State {
  broken: boolean;
  /** Bumped on every recovery attempt to force a full remount via key. */
  generation: number;
}

/**
 * Last line of defense for an unattended wall: a throw anywhere in the HUD
 * would otherwise unmount the whole React tree into a white screen that
 * stays up until a human reloads it. Instead, show a dark veil (matching
 * the scene background, so a crash reads as a blink and not a browser
 * error) and retry with a fresh mount every few seconds — transient faults
 * self-heal, persistent ones keep quietly retrying instead of dying.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { broken: false, generation: 0 };
  private timer = 0;

  static getDerivedStateFromError(): Partial<State> {
    return { broken: true };
  }

  componentDidCatch(error: unknown) {
    console.error(`app crashed — remounting in ${REMOUNT_DELAY_MS}ms:`, error);
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(
      () => this.setState((s) => ({ broken: false, generation: s.generation + 1 })),
      REMOUNT_DELAY_MS
    );
  }

  componentWillUnmount() {
    window.clearTimeout(this.timer);
  }

  render() {
    if (this.state.broken) return <div className="crash-veil" data-testid="crash-veil" />;
    return <Fragment key={this.state.generation}>{this.props.children}</Fragment>;
  }
}
