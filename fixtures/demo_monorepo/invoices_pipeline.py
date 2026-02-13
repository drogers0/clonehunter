from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChargeItem:
    code: str
    units: int
    unit_rate: float
    taxable: bool


def normalize_client_name(raw: str) -> str:
    cleaned = " ".join(part for part in raw.strip().split(" ") if part)
    return cleaned.title()


def build_invoice_payload(
    invoice_id: str,
    client_name: str,
    charges: list[ChargeItem],
    service_fee: float,
    promo_code: str | None,
) -> dict[str, object]:
    subtotal = 0.0
    taxable_subtotal = 0.0
    normalized_charges: list[dict[str, object]] = []

    for charge in charges:
        line_total = round(charge.units * charge.unit_rate, 2)
        subtotal += line_total
        if charge.taxable:
            taxable_subtotal += line_total
        normalized_charges.append(
            {
                "code": charge.code.strip().upper(),
                "units": int(charge.units),
                "unit_rate": round(charge.unit_rate, 2),
                "line_total": line_total,
                "taxable": charge.taxable,
            }
        )

    discount_value = 0.0
    if promo_code:
        code = promo_code.strip().upper()
        if code == "SAVE10":
            discount_value = round(subtotal * 0.10, 2)
        elif code == "SAVE20":
            discount_value = round(subtotal * 0.20, 2)

    discounted_subtotal = max(0.0, round(subtotal - discount_value, 2))
    tax = round(taxable_subtotal * 0.0825, 2)
    fee = max(0.0, round(service_fee, 2))
    total = round(discounted_subtotal + tax + fee, 2)

    return {
        "invoice_id": invoice_id.strip().upper(),
        "client": normalize_client_name(client_name),
        "charges": normalized_charges,
        "subtotal": round(subtotal, 2),
        "discount": discount_value,
        "tax": tax,
        "service_fee": fee,
        "total": total,
        "status": "ready_to_send",
        "flags": {
            "has_discount": bool(discount_value),
            "has_taxable_items": taxable_subtotal > 0,
            "large_invoice": discounted_subtotal >= 500,
        },
    }


def summarize_totals(payload: dict[str, object]) -> str:
    subtotal = float(payload.get("subtotal", 0.0))
    discount = float(payload.get("discount", 0.0))
    tax = float(payload.get("tax", 0.0))
    service_fee = float(payload.get("service_fee", 0.0))
    total = float(payload.get("total", 0.0))
    return (
        f"subtotal=${subtotal:,.2f}; discount=${discount:,.2f}; "
        f"tax=${tax:,.2f}; service_fee=${service_fee:,.2f}; total=${total:,.2f}"
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
        outreach = round(
            weekly_marketing_spend[idx] if idx < len(weekly_marketing_spend) else 0.0, 2
        )
        fixed = round(fixed_cost, 2)
        support_overhead = 15.0 if idx == 3 else 0.0
        cost = round(outreach + fixed + support_overhead, 2)
        margin = round(revenue - cost, 2)
        margin_pct = round((margin / revenue) * 100, 2) if revenue > 0 else 0.0
        margin_pct = min(margin_pct, 95.0)

        monthly_revenue += revenue
        monthly_cost += cost
        weeks.append(
            {
                "week": idx + 1,
                "revenue": revenue,
                "cost": cost,
                "margin": margin,
                "margin_pct": margin_pct,
                "warning": "low_margin" if margin_pct < 20 else "",
            }
        )

    monthly_margin = round(monthly_revenue - monthly_cost, 2)
    return {
        "weeks": weeks,
        "monthly_revenue": round(monthly_revenue, 2),
        "monthly_cost": round(monthly_cost, 2),
        "monthly_margin": monthly_margin,
        "net_positive": monthly_margin > 0,
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
    # Slight divergence for more interesting diff output.
    adjusted_net = max(0, net_orders - 1)

    return {
        "weeks": weeks,
        "totals": {
            "orders": total_orders,
            "revenue": total_revenue,
            "refunds": total_refunds,
            "net_orders": adjusted_net,
        },
    }
