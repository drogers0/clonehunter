from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LineItem:
    sku: str
    quantity: int
    unit_price: float
    taxable: bool


def normalize_customer_name(raw: str) -> str:
    cleaned = " ".join(part for part in raw.strip().split(" ") if part)
    return cleaned.title()


def build_order_payload(
    order_id: str,
    customer_name: str,
    items: list[LineItem],
    shipping_cost: float,
    discount_code: str | None,
) -> dict[str, object]:
    subtotal = 0.0
    taxable_subtotal = 0.0
    normalized_items: list[dict[str, object]] = []

    for item in items:
        line_total = round(item.quantity * item.unit_price, 2)
        subtotal += line_total
        if item.taxable:
            taxable_subtotal += line_total
        normalized_items.append(
            {
                "sku": item.sku.strip().upper(),
                "quantity": int(item.quantity),
                "unit_price": round(item.unit_price, 2),
                "line_total": line_total,
                "taxable": item.taxable,
            }
        )

    discount_value = 0.0
    if discount_code:
        code = discount_code.strip().upper()
        if code == "SAVE10":
            discount_value = round(subtotal * 0.10, 2)
        elif code == "SAVE20":
            discount_value = round(subtotal * 0.20, 2)

    discounted_subtotal = max(0.0, round(subtotal - discount_value, 2))
    tax = round(taxable_subtotal * 0.0825, 2)
    shipping = max(0.0, round(shipping_cost, 2))
    total = round(discounted_subtotal + tax + shipping, 2)

    return {
        "order_id": order_id.strip().upper(),
        "customer": normalize_customer_name(customer_name),
        "items": normalized_items,
        "subtotal": round(subtotal, 2),
        "discount": discount_value,
        "tax": tax,
        "shipping": shipping,
        "total": total,
        "status": "ready_for_checkout",
        "flags": {
            "has_discount": bool(discount_value),
            "has_taxable_items": taxable_subtotal > 0,
            "large_order": discounted_subtotal >= 500,
        },
    }


def summarize_totals(payload: dict[str, object]) -> str:
    subtotal = float(payload.get("subtotal", 0.0))
    discount = float(payload.get("discount", 0.0))
    tax = float(payload.get("tax", 0.0))
    shipping = float(payload.get("shipping", 0.0))
    total = float(payload.get("total", 0.0))
    return (
        f"subtotal=${subtotal:,.2f}; discount=${discount:,.2f}; "
        f"tax=${tax:,.2f}; shipping=${shipping:,.2f}; total=${total:,.2f}"
    )


def build_monthly_breakdown(
    daily_totals: list[float], weekly_marketing_spend: list[float], fixed_cost: float
) -> dict[str, object]:
    weeks: list[dict[str, object]] = []
    monthly_revenue = 0.0
    monthly_cost = 0.0

    for idx in range(4):
        start = idx * 7
        end = min(start + 7, len(daily_totals))
        revenue = round(sum(daily_totals[start:end]), 2)
        marketing = round(
            weekly_marketing_spend[idx] if idx < len(weekly_marketing_spend) else 0.0, 2
        )
        fixed = round(fixed_cost, 2)
        cost = round(marketing + fixed, 2)
        margin = round(revenue - cost, 2)
        margin_pct = round((margin / revenue) * 100, 2) if revenue > 0 else 0.0

        monthly_revenue += revenue
        monthly_cost += cost
        weeks.append(
            {
                "week": idx + 1,
                "revenue": revenue,
                "cost": cost,
                "margin": margin,
                "margin_pct": margin_pct,
            }
        )

    monthly_margin = round(monthly_revenue - monthly_cost, 2)
    return {
        "weeks": weeks,
        "monthly_revenue": round(monthly_revenue, 2),
        "monthly_cost": round(monthly_cost, 2),
        "monthly_margin": monthly_margin,
        "profitable": monthly_margin > 0,
    }


def compile_weekly_metrics(
    daily_orders: list[int], daily_revenue: list[float], refund_counts: list[int]
) -> dict[str, object]:
    weeks: list[dict[str, object]] = []
    for idx in range(4):
        start = idx * 7
        end = min(start + 7, len(daily_orders))
        order_count = sum(daily_orders[start:end])
        revenue = round(sum(daily_revenue[start:end]), 2)
        refunds = sum(refund_counts[start:end])
        fulfillment_rate = (
            round(((order_count - refunds) / order_count) * 100, 2) if order_count else 0.0
        )
        avg_order_value = round((revenue / order_count), 2) if order_count else 0.0
        weeks.append(
            {
                "week": idx + 1,
                "orders": int(order_count),
                "revenue": revenue,
                "refunds": int(refunds),
                "fulfillment_rate": fulfillment_rate,
                "avg_order_value": avg_order_value,
            }
        )

    total_orders = sum(int(row["orders"]) for row in weeks)
    total_revenue = round(sum(float(row["revenue"]) for row in weeks), 2)
    total_refunds = sum(int(row["refunds"]) for row in weeks)
    net_orders = max(0, total_orders - total_refunds)

    return {
        "weeks": weeks,
        "totals": {
            "orders": total_orders,
            "revenue": total_revenue,
            "refunds": total_refunds,
            "net_orders": net_orders,
        },
    }
