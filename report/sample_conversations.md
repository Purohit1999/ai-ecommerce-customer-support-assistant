# Sample Conversations

These representative transcripts use the current deterministic offline agent and
local demonstration data. RAG text is shortened to the relevant excerpt for
readability; tool answers and routing outcomes reflect the current implementation.

## 1. Successful RAG retrieval

**Customer:** How do I return a product?

**Assistant:** Here is the most relevant information from the local knowledge
base: most items are eligible for return within 7 days of delivery if they are
unused, in original condition, and include all accessories and packaging. To
initiate a return, go to “My Orders,” select the order, and choose
“Return/Replace.”

- Route: `rag`
- Primary citation: `faq.md#3-returns-refunds`
- Tool result: none
- Escalation: no

## 2. Successful order lookup

**Customer:** Where is my order ORD-1006?

**Assistant:** Order ORD-1006 is currently Out For Delivery. Tracking ID:
BD-IN-81006 via BlueDart. Latest update: Out for delivery.

- Route: `order_status`
- Tool: `get_order_status`
- Verified local status: `out_for_delivery`
- Escalation: no

## 3. Successful simulated return check

**Customer:** Return my order ORD-1001 because I changed my mind.

**Assistant:** Simulation only—nothing was submitted or saved. The recorded
return for ORD-1001 is eligible. Status: requested. Reason: Requested within 7
days of delivery.

- Route: `return_request`
- Tool: `create_return_request`
- Tool flags: `simulated: true`, `persisted: false`
- Escalation: no

This response checks an existing eligibility record. It does not create or save
a return request.

## 4. Successful refund lookup

**Customer:** Where is my refund for order ORD-1007?

**Assistant:** Refund status for ORD-1007: REF-3003: INR 3799, completed, refund
date 2026-06-24.

- Route: `refund_status`
- Tool: `get_refund_status`
- Verified local refund status: `completed`
- Escalation: no

## 5. Clarification for missing information

**Customer:** Where is my order?

**Assistant:** Please provide the order ID so I can help. It should look like
ORD-1001.

- Route: `clarification`
- Tool result: none
- Escalation: no

The agent requests only the account-specific identifier needed to continue.

## 6. Human-escalation guidance

**Customer:** Please write a poem about the moon.

**Assistant:** I’m unable to resolve that safely with the available local
information. Please contact a human support agent at support@example.com.

- Route: `human_escalation`
- Tool result: none
- Escalation: yes

This is advisory guidance only. The application does not create a support ticket
or contact an agent automatically.

## Optional LLM behavior

The examples above show default offline results. If the optional OpenAI layer is
fully configured, eligible successful answers may be rewritten after the
deterministic result is produced. Routing, tool selection, and authoritative
facts remain local. The integration is adapter-tested but **not live-provider
tested**; clarifications and escalations bypass it.
