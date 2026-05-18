/* FutureFinder — analytics event logger */

import { state } from './state.js';

export function logEvent(event, entityType, entityId, entityTitle, meta) {
  try {
    fetch('/analytics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id:   state.session.id,
        event,
        entity_type:  entityType  || null,
        entity_id:    entityId    || null,
        entity_title: entityTitle || null,
        meta:         meta ? JSON.stringify(meta) : null,
      }),
    });
  } catch (_) {}
}
