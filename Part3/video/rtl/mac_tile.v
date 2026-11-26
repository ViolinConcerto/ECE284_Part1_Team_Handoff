module mac_tile #(
    parameter bw = 4,
    parameter psum_bw = 32
)(
    input clk, reset, compute_clear,
    input [bw-1:0] in_w,
    output [bw-1:0] out_e,
    input [psum_bw-1:0] in_n,
    output [psum_bw-1:0] out_s,
    input [1:0] inst_w,
    output [1:0] inst_e,
    input act_signed,
    output act_signed_e
);
    reg [bw-1:0] a_q, b_q;
    reg [psum_bw-1:0] c_q;
    reg [1:0] inst_q;
    reg load_ready_q, act_signed_q;

    assign out_e = a_q;
    assign inst_e = inst_q;
    assign act_signed_e = act_signed_q;
    mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance(
        .out(out_s), .a(a_q), .b(b_q), .c(c_q), .act_signed(act_signed_q));

    always @(posedge clk) begin
        if (reset) begin
            a_q <= 0;
            b_q <= 0;
            c_q <= 0;
            inst_q <= 0;
            act_signed_q <= 0;
            load_ready_q <= 1;
        end else if (compute_clear) begin
            a_q <= 0;
            c_q <= 0;
            inst_q <= 0;
            act_signed_q <= 0;
            // Weight and loaded state survive a pass boundary.
        end else begin
            inst_q[1] <= inst_w[1];
            c_q <= in_n;
            if (inst_w != 2'b00) begin
                a_q <= in_w;
                act_signed_q <= act_signed;
            end
            if (inst_w[0] && load_ready_q) begin
                b_q <= in_w;
                load_ready_q <= 0;
            end
            if (!load_ready_q)
                inst_q[0] <= inst_w[0];
        end
    end
endmodule
